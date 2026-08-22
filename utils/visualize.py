import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.prompt_utils import SYSTEM_PROMPT
from utils.runtime import autocast_ctx
from utils.vlm_inputs import _get_pad_token_id, pad_and_stack_vlm_inputs


def _save_last_batch_visualizations(
    *,
    out_dir: Path,
    epoch_id: int,
    images: List[Image.Image],
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    alpha: float = 0.45,
    max_items: int = 4,
    vis_prefix: str = "vis",
) -> None:
    """Save segmentation visualizations for the last batch of an epoch.

    Writes per-sample:
      - <idx>_img.png
      - <idx>_gt.png
      - <idx>_pred.png
      - <idx>_overlay.png
    """
    vis_dir = out_dir / f"{vis_prefix}_epoch{epoch_id}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pred_probs = torch.sigmoid(pred_logits.detach().cpu())  # [B,1,H,W]
    gt = gt_mask.detach().cpu()  # [B,1,H,W]

    b = min(len(images), int(pred_probs.shape[0]), max_items)
    for i in range(b):
        img = images[i]
        w, h = img.size

        pred_i = pred_probs[i, 0].numpy()
        gt_i = gt[i, 0].numpy()

        # resize to original image size for visualization
        pred_resized = cv2.resize(pred_i, (w, h), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt_i, (w, h), interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_resized >= float(threshold)).astype(np.uint8) * 255
        gt_u8 = (gt_resized >= 0.5).astype(np.uint8) * 255

        img_rgb = np.array(img.convert("RGB"), dtype=np.uint8)

        # overlay predicted mask in red
        overlay = img_rgb.copy()
        red = np.zeros_like(img_rgb)
        red[..., 0] = 255
        m = (pred_bin > 0)[..., None]
        overlay = np.where(m, (overlay * (1.0 - alpha) + red * alpha).astype(np.uint8), overlay)

        Image.fromarray(img_rgb).save(vis_dir / f"{i:02d}_img.png")
        Image.fromarray(gt_u8).save(vis_dir / f"{i:02d}_gt.png")
        Image.fromarray(pred_bin).save(vis_dir / f"{i:02d}_pred.png")
        Image.fromarray(overlay).save(vis_dir / f"{i:02d}_overlay.png")


def _decode_tokens(processor: Any, token_ids: torch.Tensor) -> List[str]:
    """Decode token ids to text, robust across processor versions."""
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode(token_ids, skip_special_tokens=True)
        except Exception:
            pass
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "batch_decode"):
        return tok.batch_decode(token_ids, skip_special_tokens=True)
    # Last resort
    return [str(x.tolist()) for x in token_ids]


@torch.no_grad()
def _generate_vlm_texts(
    *,
    processor: Any,
    vlm: Any,
    device: torch.device,
    images: List[Image.Image],
    best_prompts: List[str],
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
) -> List[str]:
    """Generate VLM texts for a batch of images and prompts."""
    if len(images) == 0:
        return []

    messages_batch = []
    for img, prompt in zip(images, best_prompts):
        messages_batch.append(
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": f"Locate and assess {prompt} <seg>."}, {"type": "image", "image": img}]},
            ]
        )

    vlm_inputs_list = [
        processor.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        for msgs in messages_batch
    ]

    pad_id = _get_pad_token_id(processor)
    model_kwargs = pad_and_stack_vlm_inputs(vlm_inputs_list, device, pad_id)

    was_training = bool(getattr(vlm, "training", False))
    try:
        vlm.eval()
        with autocast_ctx(device):
            gen = vlm.generate(
                **model_kwargs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                num_beams=int(num_beams),
            )
    finally:
        if was_training:
            vlm.train()

    prompt_lens = model_kwargs["attention_mask"].sum(dim=1).tolist()
    texts: List[str] = []
    for i in range(gen.shape[0]):
        pl = int(prompt_lens[i])
        new_ids = gen[i, pl:]
        texts.append(_decode_tokens(processor, new_ids.unsqueeze(0))[0].strip())
    return texts


@torch.no_grad()
def _save_last_batch_vlm_texts(
    *,
    out_dir: Path,
    epoch_id: int,
    processor: Any,
    vlm: Any,
    device: torch.device,
    images: List[Image.Image],
    best_prompts: List[str],
    gt_texts: List[str],
    image_paths: Optional[List[str]] = None,
    vis_prefix: str = "vis",
    max_items: int = 4,
    out_filename: str = "last_batch_texts.jsonl",
    max_new_tokens: int = 96,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
) -> None:
    """Generate VLM text on the last batch and save alongside visualizations."""
    vis_dir = out_dir / f"{vis_prefix}_epoch{epoch_id}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    b = min(len(images), len(best_prompts), len(gt_texts), int(max_items))
    images = images[:b]
    best_prompts = best_prompts[:b]
    gt_texts = gt_texts[:b]
    if image_paths is not None:
        image_paths = image_paths[:b]

    texts = _generate_vlm_texts(
        processor=processor,
        vlm=vlm,
        device=device,
        images=images,
        best_prompts=best_prompts,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
    )

    out_path = vis_dir / out_filename
    with out_path.open("w", encoding="utf-8") as f:
        for i, (prompt, pred, gt) in enumerate(zip(best_prompts, texts, gt_texts)):
            rec = {
                "index": i,
                "prompt": prompt,
                "vlm_text": pred,
                "gt_text": gt,
            }
            if image_paths is not None and i < len(image_paths):
                rec["image_path"] = image_paths[i]
            formatted_json = json.dumps(rec, ensure_ascii=False, indent=2)
            f.write(formatted_json + "\n" + "-" * 50 + "\n")
