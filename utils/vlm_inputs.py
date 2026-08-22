from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn.functional as F


def _get_pad_token_id(processor: Any) -> int:
    """Get pad_token_id from processor or its tokenizer."""
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is not None and hasattr(obj, "pad_token_id") and obj.pad_token_id is not None:
            return int(obj.pad_token_id)
    return 0


def pad_and_stack_vlm_inputs(
    vlm_inputs_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
    pad_token_id: int = 0,
    labels_list: Optional[List[torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Pad variable-length per-sample VLM inputs and stack into a batch.

    Left-pads input_ids/attention_mask so that causal generation works correctly.
    Optionally pads labels with -100 (ignored by CE loss).
    """
    max_len = max(d["input_ids"].shape[1] for d in vlm_inputs_list)

    padded_ids = []
    padded_mask = []
    padded_labels = [] if labels_list is not None else None

    for i, d in enumerate(vlm_inputs_list):
        seq_len = d["input_ids"].shape[1]
        pad_len = max_len - seq_len

        ids = F.pad(d["input_ids"], (pad_len, 0), value=pad_token_id)
        mask = F.pad(
            d.get("attention_mask", torch.ones_like(d["input_ids"])),
            (pad_len, 0),
            value=0,
        )
        padded_ids.append(ids)
        padded_mask.append(mask)

        if labels_list is not None:
            lbl = F.pad(labels_list[i], (pad_len, 0), value=-100)
            padded_labels.append(lbl)

    result: Dict[str, torch.Tensor] = {
        "input_ids": torch.cat(padded_ids, dim=0).to(device),
        "attention_mask": torch.cat(padded_mask, dim=0).to(device),
    }

    for k in ("pixel_values", "images", "vision_pixel_values"):
        if k in vlm_inputs_list[0]:
            result[k] = torch.cat([d[k] for d in vlm_inputs_list], dim=0).to(device)
            break

    if padded_labels is not None:
        result["labels"] = torch.cat(padded_labels, dim=0).to(device)

    return result


def build_teacher_forcing_inputs(
    processor: Any,
    images: List[Image.Image],
    prompts: List[str],
    target_texts: List[str],
    system_prompt: str,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    """Build per-sample VLM inputs + labels for teacher-forcing CE loss.

    For each sample:
    1. Tokenize the full conversation (system + user + assistant with target_text).
    2. Tokenize the prompt only (system + user + generation prompt) to get prompt length.
    3. Labels = input_ids with -100 for prompt positions.

    Returns (vlm_inputs_list, labels_list).
    """
    vlm_inputs_list: List[Dict[str, torch.Tensor]] = []
    labels_list: List[torch.Tensor] = []

    for img, prompt, target in zip(images, prompts, target_texts):
        # Full conversation with assistant response (unified <seg> template)
        messages_full = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"Locate and assess {prompt} <seg>."}, {"type": "image", "image": img}]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        full_inputs = processor.apply_chat_template(
            messages_full,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Prompt only (to determine where assistant tokens start)
        messages_prompt = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"Locate and assess {prompt} <seg>."}, {"type": "image", "image": img}]},
        ]
        prompt_only = processor.apply_chat_template(
            messages_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_len = prompt_only["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100  # mask prompt tokens

        vlm_inputs_list.append(full_inputs)
        labels_list.append(labels)

    return vlm_inputs_list, labels_list


def build_anatopave_teacher_forcing_inputs(
    processor: Any,
    images: List[Image.Image],
    prompts: List[str],
    target_texts: List[str],
    system_prompt: str,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    """Build ANATOPAVE inputs with one explicit evidence placeholder.

    The projected Composer context replaces ``<evidence>`` at the embedding
    layer.  Keeping this token in the normal chat template lets the same input
    representation work for teacher forcing and generation.
    """
    inputs: List[Dict[str, torch.Tensor]] = []
    labels_list: List[torch.Tensor] = []
    for image, prompt, target in zip(images, prompts, target_texts):
        user_text = (
            f"Current evidence: <evidence>. "
            f"Locate and assess the current view {prompt} <seg>."
        )
        common = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            },
        ]
        full = processor.apply_chat_template(
            common + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_only = processor.apply_chat_template(
            common,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = full["input_ids"].clone()
        labels[:, : prompt_only["input_ids"].shape[1]] = -100
        inputs.append(full)
        labels_list.append(labels)
    return inputs, labels_list


def build_anatopave_generation_inputs(
    processor: Any,
    images: List[Image.Image],
    prompts: List[str],
    system_prompt: str,
) -> List[Dict[str, torch.Tensor]]:
    """Build generation prompts with the same evidence/seg placeholders."""
    output: List[Dict[str, torch.Tensor]] = []
    for image, prompt in zip(images, prompts):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Current evidence: <evidence>. "
                            f"Locate and assess the current view {prompt} <seg>."
                        ),
                    },
                    {"type": "image", "image": image},
                ],
            },
        ]
        output.append(
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )
    return output
