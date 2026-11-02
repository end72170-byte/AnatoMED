"""
Video inference with FetalCLIP-triggered FSM and buffered keyframe selection.

Example:
python inference.py \
  --video_path /data/clip-dataset/all/case/normal/high/video.avi \
  --checkpoint_path /checkpoints/anatomed.pt \
  --fetalclip_weights /checkpoints/fetalclip.pt \
  --out_json runs/video_events.json \
  --save_dir runs/video_events
  --enter_threshold 0.20
  --exit_threshold 0.15
  --margin_threshold 0.02
  --enter_streak 3
  --min_state_frames 4

"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator

import open_clip
from open_clip.factory import _MODEL_CONFIGS

from utils.models import (
    FetalCLIPSegFeatureExtractor,
    VLMConditionedSegDecoder,
    extract_seg_token_feature,
    extract_vlm_vision_tokens,
)
from utils.prompt_utils import SYSTEM_PROMPT, build_prompt_text_features
from utils.runtime import _configure_warnings, autocast_ctx
from utils.visualize import _generate_vlm_texts
from utils.vlm_inputs import _get_pad_token_id, pad_and_stack_vlm_inputs


DEFAULT_KEYFRAME_PROMPTS = [
    "fetal ultrasound, fetal eyes.",
    "fetal ultrasound, upper alveolar ridge.",
    "fetal ultrasound, intact hard palate.",
    "fetal ultrasound, lower alveolar ridge.",
    "fetal ultrasound, cleft hard palate.",
]

CLASS_LABELS = [
    "EYES",
    "UPPER_ALVEOLAR",
    "INTACT_HARD_PALATE",
    "LOWER_ALVEOLAR",
    "CLEFT_HARD_PALATE",
]
CLASS_TO_PROMPT = dict(zip(CLASS_LABELS, DEFAULT_KEYFRAME_PROMPTS))
ALLOWED_NEXT = {
    "INIT": ["EYES", "UPPER_ALVEOLAR", "LOWER_ALVEOLAR"],
    "EYES": ["EYES", "UPPER_ALVEOLAR"],
    "UPPER_ALVEOLAR": ["UPPER_ALVEOLAR", "LOWER_ALVEOLAR", "INTACT_HARD_PALATE", "CLEFT_HARD_PALATE"],
    "LOWER_ALVEOLAR": ["LOWER_ALVEOLAR", "INTACT_HARD_PALATE", "CLEFT_HARD_PALATE"],
    # In real scans a brief hard-palate glimpse can appear before the lower
    # alveolar ridge is stabilized, so allow a fallback from HARD to LOWER.
    # Also allow intact/cleft hard-palate states to swap so the FSM can absorb
    # within-palate diagnostic ambiguity instead of getting stuck on one label.
    "INTACT_HARD_PALATE": ["INTACT_HARD_PALATE", "CLEFT_HARD_PALATE", "LOWER_ALVEOLAR"],
    "CLEFT_HARD_PALATE": ["CLEFT_HARD_PALATE", "INTACT_HARD_PALATE", "LOWER_ALVEOLAR"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime-style video inference with FSM")
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--save_dir", type=str, default="")

    p.add_argument("--fetalclip_config", type=str, default="model/FetalCLIP_config.json")
    p.add_argument("--fetalclip_weights", type=str, required=True)
    p.add_argument("--vlm_model_id", type=str, default="google/medgemma-4b-it")
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_seg_token", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gen_max_new_tokens", type=int, default=256)
    p.add_argument("--gen_do_sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gen_temperature", type=float, default=0.0)
    p.add_argument("--gen_top_p", type=float, default=1.0)
    p.add_argument("--gen_num_beams", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--frame_stride", type=int, default=1, help="Process every N-th frame.")
    p.add_argument("--max_frames", type=int, default=0, help="0 means process the whole video.")

    p.add_argument("--ema_alpha", type=float, default=0.5)
    p.add_argument("--enter_threshold", type=float, default=0.33)
    p.add_argument("--exit_threshold", type=float, default=0.24)
    p.add_argument("--margin_threshold", type=float, default=0.05)
    p.add_argument("--enter_streak", type=int, default=4)
    p.add_argument("--exit_patience", type=int, default=3)
    p.add_argument("--cooldown_frames", type=int, default=40)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--min_state_frames", type=int, default=4)
    p.add_argument("--max_state_frames", type=int, default=36, help="Force-finalize a state after this many accepted frames.")
    p.add_argument("--sharpness_weight", type=float, default=0.03)
    p.add_argument("--margin_weight", type=float, default=0.40)
    p.add_argument("--motion_weight", type=float, default=0.20)
    p.add_argument("--palate_cleft_ratio_thr", type=float, default=0.2, help="Video-level postprocess threshold on palate_cleft_ratio for abnormal diagnosis.")
    return p.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def load_fetalclip(args: argparse.Namespace, device: torch.device, log) -> Tuple[Any, Any, Any, torch.Tensor]:
    with open(args.fetalclip_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _MODEL_CONFIGS["FetalCLIP"] = cfg

    fetalclip_model, _, fetalclip_preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=None
    )
    if args.fetalclip_weights:
        log(f"Loading FetalCLIP weights from {args.fetalclip_weights}...")
        checkpoint = torch.load(args.fetalclip_weights, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        fetalclip_model.load_state_dict(state_dict, strict=True)

    fetalclip_tokenizer = open_clip.get_tokenizer("FetalCLIP")
    fetalclip_model.eval().to(device)
    for p in fetalclip_model.parameters():
        p.requires_grad = False

    prompt_text_features = build_prompt_text_features(
        fetalclip_model,
        fetalclip_tokenizer,
        device,
        DEFAULT_KEYFRAME_PROMPTS,
    )
    return fetalclip_model, fetalclip_preprocess, fetalclip_tokenizer, prompt_text_features


def load_vlm_and_heads(args: argparse.Namespace, device: torch.device, log) -> Tuple[Any, Any, Any, int]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as e:
        raise RuntimeError("Missing transformers. Install deps via requirements.txt") from e

    try:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    except Exception as e:
        raise RuntimeError("LoRA evaluation requires peft. Install: pip install peft accelerate") from e

    def _load_from_pretrained_with_fallback(load_fn, *, what: str, **kwargs):
        try:
            return load_fn(**kwargs)
        except Exception as e:
            if kwargs.get("local_files_only"):
                raise
            log(f"[warn] Failed to load {what} online ({e}); retrying with local_files_only=True")
            kwargs["local_files_only"] = True
            return load_fn(**kwargs)

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        vlm = _load_from_pretrained_with_fallback(
            AutoModelForImageTextToText.from_pretrained,
            what="VLM model",
            pretrained_model_name_or_path=args.vlm_model_id,
            dtype=model_dtype,
            local_files_only=bool(args.local_files_only),
        ).to(device)
    except TypeError:
        vlm = _load_from_pretrained_with_fallback(
            AutoModelForImageTextToText.from_pretrained,
            what="VLM model",
            pretrained_model_name_or_path=args.vlm_model_id,
            torch_dtype=model_dtype,
            local_files_only=bool(args.local_files_only),
        ).to(device)

    processor = _load_from_pretrained_with_fallback(
        AutoProcessor.from_pretrained,
        what="VLM processor",
        pretrained_model_name_or_path=args.vlm_model_id,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )

    processor.tokenizer.add_tokens(["<seg>"])
    vlm.resize_token_embeddings(len(processor.tokenizer))
    seg_token_id = processor.tokenizer.convert_tokens_to_ids("<seg>")
    log(f"[seg] Added <seg> token, id={seg_token_id}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    vlm = get_peft_model(vlm, lora_cfg)

    dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": "Locate and assess fetal ultrasound. <seg>."}, {"type": "image", "image": dummy_img}]},
    ]
    dummy_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
    with torch.no_grad():
        dummy_out = vlm(**dummy_inputs, output_hidden_states=True, return_dict=True)
    d_vlm = int(extract_vlm_vision_tokens(dummy_out).shape[-1])

    seg_head = VLMConditionedSegDecoder(
        in_channels=1024,
        d_vlm=d_vlm,
        num_classes=1,
        num_heads=8,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if "seg_head" not in checkpoint:
        raise RuntimeError(f"Checkpoint missing seg_head: {args.checkpoint_path}")
    seg_head.load_state_dict(checkpoint["seg_head"], strict=True)
    if "vlm_lora" in checkpoint:
        set_peft_model_state_dict(vlm, checkpoint["vlm_lora"])
    elif "vlm" in checkpoint:
        vlm.load_state_dict(checkpoint["vlm"], strict=False)
    else:
        raise RuntimeError(f"Checkpoint missing vlm/vlm_lora weights: {args.checkpoint_path}")

    vlm.eval()
    seg_head.eval()
    return vlm, processor, seg_head, seg_token_id


def build_generation_inputs(
    *,
    processor: Any,
    device: torch.device,
    images: List[Image.Image],
    best_prompts: List[str],
) -> Dict[str, torch.Tensor]:
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
    return pad_and_stack_vlm_inputs(vlm_inputs_list, device, pad_id)


def pil_from_bgr(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def compute_sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def compute_motion(prev_bgr: Optional[np.ndarray], curr_bgr: np.ndarray) -> float:
    if prev_bgr is None:
        return 0.0
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean() / 255.0)


@dataclass
class CandidateFrame:
    frame_idx: int
    timestamp_sec: float
    label: str
    prompt: str
    image_pil: Image.Image
    score: float
    margin: float
    sharpness: float
    motion: float
    rank_score: float
    scores: Dict[str, float]


@dataclass
class StateBuffer:
    label: str
    topk: int
    start_frame: int
    start_time_sec: float
    candidates: List[CandidateFrame] = field(default_factory=list)
    observed_frames: int = 0
    best_score: float = -1e9

    def add(self, item: CandidateFrame) -> None:
        self.observed_frames += 1
        self.best_score = max(self.best_score, item.score)
        self.candidates.append(item)
        self.candidates.sort(key=lambda x: x.rank_score, reverse=True)
        if len(self.candidates) > self.topk:
            self.candidates = self.candidates[: self.topk]

    def best(self) -> Optional[CandidateFrame]:
        return self.candidates[0] if self.candidates else None


class FSMController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.current_state = "INIT"
        self.current_buffer: Optional[StateBuffer] = None
        self.ema_scores = np.zeros(len(CLASS_LABELS), dtype=np.float32)
        self.prev_top_idx: Optional[int] = None
        self.top_streak = 0
        self.missing_count = 0
        self.last_confirmed_state = "INIT"
        self.last_finalized_frame: Dict[str, int] = {label: -10**9 for label in CLASS_LABELS}

    def _eligible_transition(self, label: str, score: float, margin: float, streak: int, frame_idx: int) -> bool:
        if score < float(self.args.enter_threshold):
            return False
        if margin < float(self.args.margin_threshold):
            return False
        if streak < int(self.args.enter_streak):
            return False
        if frame_idx - self.last_finalized_frame[label] < int(self.args.cooldown_frames):
            return False
        return True

    def _start_state(self, label: str, frame_idx: int, timestamp_sec: float) -> None:
        self.current_state = label
        self.current_buffer = StateBuffer(
            label=label,
            topk=int(self.args.topk),
            start_frame=frame_idx,
            start_time_sec=timestamp_sec,
        )
        self.missing_count = 0

    def _allowed_idle_labels(self) -> List[str]:
        allowed = list(ALLOWED_NEXT.get(self.last_confirmed_state, ALLOWED_NEXT["INIT"]))
        if self.last_confirmed_state == "INIT":
            return allowed
        return [label for label in allowed if label != self.last_confirmed_state]

    def _finalize_current(self) -> Optional[StateBuffer]:
        buf = self.current_buffer
        if buf is None:
            return None
        if buf.observed_frames < int(self.args.min_state_frames) or buf.best() is None:
            self.current_buffer = None
            self.current_state = "INIT" if self.current_state == "EYES" else self.current_state
            return None
        self.last_finalized_frame[buf.label] = buf.best().frame_idx
        self.last_confirmed_state = buf.label
        self.current_buffer = None
        return buf

    def update(
        self,
        *,
        frame_idx: int,
        timestamp_sec: float,
        frame_pil: Image.Image,
        raw_scores: Dict[str, float],
        sharpness: float,
        motion: float,
    ) -> List[StateBuffer]:
        score_vec = np.array([raw_scores[label] for label in CLASS_LABELS], dtype=np.float32)
        alpha = float(self.args.ema_alpha)
        self.ema_scores = alpha * self.ema_scores + (1.0 - alpha) * score_vec

        top_idx = int(np.argmax(self.ema_scores))
        sorted_scores = np.sort(self.ema_scores)
        top_score = float(self.ema_scores[top_idx])
        second_score = float(sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
        margin = top_score - second_score
        top_label = CLASS_LABELS[top_idx]

        if self.prev_top_idx == top_idx:
            self.top_streak += 1
        else:
            self.top_streak = 1
            self.prev_top_idx = top_idx

        rank_score = (
            float(score_vec[top_idx])
            + float(self.args.margin_weight) * margin
            + float(self.args.sharpness_weight) * math.log1p(max(sharpness, 0.0))
            - float(self.args.motion_weight) * motion
        )
        candidate = CandidateFrame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            label=top_label,
            prompt=CLASS_TO_PROMPT[top_label],
            image_pil=frame_pil.copy(),
            score=float(score_vec[top_idx]),
            margin=margin,
            sharpness=sharpness,
            motion=motion,
            rank_score=rank_score,
            scores={k: float(v) for k, v in raw_scores.items()},
        )

        finalized: List[StateBuffer] = []

        if self.current_buffer is None:
            if self._eligible_transition(top_label, top_score, margin, self.top_streak, frame_idx):
                if top_label in self._allowed_idle_labels():
                    self._start_state(top_label, frame_idx, timestamp_sec)
                    self.current_buffer.add(candidate)
            return finalized

        current_label = self.current_buffer.label
        current_idx = CLASS_LABELS.index(current_label)
        current_score = float(self.ema_scores[current_idx])

        if top_label == current_label:
            self.current_buffer.add(candidate)
            self.missing_count = 0
        elif current_score >= float(self.args.exit_threshold):
            # Keep the state alive via hysteresis, but do not let another label
            # overwrite the current state's candidate pool.
            self.missing_count = 0
        else:
            self.missing_count += 1

        allowed = ALLOWED_NEXT.get(current_label, [current_label])
        if top_label != current_label and top_label in allowed and self._eligible_transition(top_label, top_score, margin, self.top_streak, frame_idx):
            finished = self._finalize_current()
            if finished is not None:
                finalized.append(finished)
            self._start_state(top_label, frame_idx, timestamp_sec)
            self.current_buffer.add(candidate)
            return finalized

        if self.current_buffer is not None and self.current_buffer.observed_frames >= int(self.args.max_state_frames):
            finished = self._finalize_current()
            if finished is not None:
                finalized.append(finished)
            self.current_state = "INIT"
            self.missing_count = 0
            return finalized

        if self.missing_count >= int(self.args.exit_patience):
            finished = self._finalize_current()
            if finished is not None:
                finalized.append(finished)
            self.current_state = "INIT"
            self.missing_count = 0

        return finalized

    def flush(self) -> List[StateBuffer]:
        finished = self._finalize_current()
        self.current_state = "INIT"
        self.missing_count = 0
        return [finished] if finished is not None else []


@torch.no_grad()
def score_frame(
    fetalclip_model: Any,
    fetalclip_preprocess: Any,
    prompt_text_features: torch.Tensor,
    device: torch.device,
    image_pil: Image.Image,
) -> Dict[str, float]:
    img_t = fetalclip_preprocess(image_pil)
    img_t = torch.stack([img_t]).to(device)
    with autocast_ctx(device):
        img_feat = fetalclip_model.encode_image(img_t)
    img_feat = l2_normalize(img_feat, dim=-1)
    sims = (img_feat @ prompt_text_features.t()).squeeze(0).detach().float().cpu().tolist()
    return {label: float(score) for label, score in zip(CLASS_LABELS, sims)}


@torch.no_grad()
def infer_event(
    *,
    candidate: CandidateFrame,
    state_label: str,
    vlm: Any,
    processor: Any,
    fetalclip_preprocess: Any,
    seg_feature_extractor: FetalCLIPSegFeatureExtractor,
    seg_head: Any,
    seg_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Optional[Path],
    event_idx: int,
) -> Dict[str, Any]:
    image = candidate.image_pil
    prompt = CLASS_TO_PROMPT[state_label]

    model_kwargs = build_generation_inputs(
        processor=processor,
        device=device,
        images=[image],
        best_prompts=[prompt],
    )

    fc_imgs = torch.stack([fetalclip_preprocess(image)]).to(device)
    with autocast_ctx(device):
        fc_features, _ = seg_feature_extractor(fc_imgs)
        out = vlm(**model_kwargs, output_hidden_states=True, return_dict=True)
        vlm_hidden = out.hidden_states[-1]
        if args.use_seg_token and seg_token_id >= 0:
            vlm_hidden = extract_seg_token_feature(vlm_hidden, model_kwargs["input_ids"], seg_token_id)
        seg_logits = seg_head(fc_features, vlm_hidden, out_hw=(224, 224))

    pred_text = _generate_vlm_texts(
        processor=processor,
        vlm=vlm,
        device=device,
        images=[image],
        best_prompts=[prompt],
        max_new_tokens=int(args.gen_max_new_tokens),
        do_sample=bool(args.gen_do_sample),
        temperature=float(args.gen_temperature),
        top_p=float(args.gen_top_p),
        num_beams=int(args.gen_num_beams),
    )[0]

    pred_prob = torch.sigmoid(seg_logits[0, 0]).detach().float().cpu().numpy()
    pred_mask = (pred_prob >= 0.5).astype(np.uint8) * 255
    pred_score = float(pred_prob.max())

    record: Dict[str, Any] = {
        "event_index": event_idx,
        "label": state_label,
        "prompt": prompt,
        "frame_idx": candidate.frame_idx,
        "timestamp_sec": candidate.timestamp_sec,
        "clip_score": candidate.score,
        "clip_margin": candidate.margin,
        "sharpness": candidate.sharpness,
        "motion": candidate.motion,
        "text": pred_text,
        "mask_score": pred_score,
        "class_scores": candidate.scores,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"event_{event_idx:02d}_{state_label.lower()}_f{candidate.frame_idx:06d}"
        img_path = save_dir / f"{stem}_img.png"
        mask_path = save_dir / f"{stem}_mask.png"
        overlay_path = save_dir / f"{stem}_overlay.png"

        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_resized = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = image_rgb.copy()
        red = np.zeros_like(image_rgb)
        red[..., 0] = 255
        m = (mask_resized > 0)[..., None]
        overlay = np.where(m, (overlay * 0.55 + red * 0.45).astype(np.uint8), overlay)

        Image.fromarray(image_rgb).save(img_path)
        Image.fromarray(mask_resized).save(mask_path)
        Image.fromarray(overlay).save(overlay_path)

        record["image_path"] = str(img_path)
        record["mask_path"] = str(mask_path)
        record["overlay_path"] = str(overlay_path)

    return record


def merge_adjacent_same_label_events(events: List[Dict[str, Any]], gap_frames: int) -> List[Dict[str, Any]]:
    if not events:
        return []

    merged: List[Dict[str, Any]] = [dict(events[0])]
    for event in events[1:]:
        prev = merged[-1]
        gap = int(event.get("segment_start_frame", event["frame_idx"])) - int(prev.get("frame_idx", prev.get("segment_start_frame", 0)))
        if event["label"] == prev["label"] and gap <= int(gap_frames):
            prev_end = int(prev.get("frame_idx", prev.get("segment_start_frame", 0)))
            curr_end = int(event.get("frame_idx", event.get("segment_start_frame", 0)))
            if float(event.get("clip_score", -1.0)) > float(prev.get("clip_score", -1.0)):
                keep_segment_start = prev.get("segment_start_frame", prev.get("frame_idx"))
                keep_segment_start_sec = prev.get("segment_start_sec", prev.get("timestamp_sec"))
                prev.clear()
                prev.update(event)
                prev["segment_start_frame"] = keep_segment_start
                prev["segment_start_sec"] = keep_segment_start_sec
            prev["segment_observed_frames"] = int(prev.get("segment_observed_frames", 0)) + int(event.get("segment_observed_frames", 0))
            prev["segment_best_score"] = max(float(prev.get("segment_best_score", -1.0)), float(event.get("segment_best_score", -1.0)))
        else:
            merged.append(dict(event))

    for idx, event in enumerate(merged):
        event["event_index"] = idx
    return merged


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(cpu=bool(args.cpu))
    log = accelerator.print
    set_random_seed(int(args.seed))
    _configure_warnings(suppress=bool(args.suppress_warnings))
    device = accelerator.device

    fetalclip_model, fetalclip_preprocess, _, prompt_text_features = load_fetalclip(args, device, log)
    vlm, processor, seg_head, seg_token_id = load_vlm_and_heads(args, device, log)
    seg_feature_extractor = FetalCLIPSegFeatureExtractor(fetalclip_model.visual)
    seg_feature_extractor.eval()
    seg_head.eval()
    vlm.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log(f"[video] path={args.video_path} fps={fps:.3f} total_frames={total_frames}")

    fsm = FSMController(args)
    save_dir = Path(args.save_dir) if args.save_dir else None
    events: List[Dict[str, Any]] = []
    frame_top1_counts: Counter[str] = Counter()
    prev_bgr: Optional[np.ndarray] = None
    frame_idx = -1
    processed = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if int(args.frame_stride) > 1 and (frame_idx % int(args.frame_stride) != 0):
            continue
        processed += 1
        if int(args.max_frames) > 0 and processed > int(args.max_frames):
            break

        timestamp_sec = frame_idx / fps
        frame_pil = pil_from_bgr(frame_bgr)
        sharpness = compute_sharpness(frame_bgr)
        motion = compute_motion(prev_bgr, frame_bgr)
        raw_scores = score_frame(
            fetalclip_model=fetalclip_model,
            fetalclip_preprocess=fetalclip_preprocess,
            prompt_text_features=prompt_text_features,
            device=device,
            image_pil=frame_pil,
        )
        top1_label = max(raw_scores.items(), key=lambda kv: kv[1])[0]
        frame_top1_counts[top1_label] += 1

        finalized_buffers = fsm.update(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            frame_pil=frame_pil,
            raw_scores=raw_scores,
            sharpness=sharpness,
            motion=motion,
        )
        for buf in finalized_buffers:
            best = buf.best()
            if best is None:
                continue
            event = infer_event(
                candidate=best,
                state_label=buf.label,
                vlm=vlm,
                processor=processor,
                fetalclip_preprocess=fetalclip_preprocess,
                seg_feature_extractor=seg_feature_extractor,
                seg_head=seg_head,
                seg_token_id=seg_token_id,
                device=device,
                args=args,
                save_dir=save_dir,
                event_idx=len(events),
            )
            event["segment_start_frame"] = buf.start_frame
            event["segment_start_sec"] = buf.start_time_sec
            event["segment_observed_frames"] = buf.observed_frames
            event["segment_best_score"] = buf.best_score
            events.append(event)
            log(
                f"[event] idx={event['event_index']} label={event['label']} frame={event['frame_idx']} "
                f"t={event['timestamp_sec']:.2f}s clip={event['clip_score']:.4f}"
            )

        prev_bgr = frame_bgr.copy()

    cap.release()

    for buf in fsm.flush():
        best = buf.best()
        if best is None:
            continue
        event = infer_event(
            candidate=best,
            state_label=buf.label,
            vlm=vlm,
            processor=processor,
            fetalclip_preprocess=fetalclip_preprocess,
            seg_feature_extractor=seg_feature_extractor,
            seg_head=seg_head,
            seg_token_id=seg_token_id,
            device=device,
            args=args,
            save_dir=save_dir,
            event_idx=len(events),
        )
        event["segment_start_frame"] = buf.start_frame
        event["segment_start_sec"] = buf.start_time_sec
        event["segment_observed_frames"] = buf.observed_frames
        event["segment_best_score"] = buf.best_score
        events.append(event)

    events = merge_adjacent_same_label_events(events, gap_frames=int(args.cooldown_frames))

    cleft_count = int(frame_top1_counts.get("CLEFT_HARD_PALATE", 0))
    intact_count = int(frame_top1_counts.get("INTACT_HARD_PALATE", 0))
    palate_total = int(cleft_count + intact_count)
    palate_cleft_ratio = float(cleft_count / palate_total) if palate_total > 0 else 0.0
    cleft_ratio_all_frames = float(cleft_count / processed) if processed > 0 else 0.0
    diagnose = "abnormal" if palate_cleft_ratio >= float(args.palate_cleft_ratio_thr) else "normal"
    frame_top1_prompt_counts = {
        CLASS_TO_PROMPT[label]: int(frame_top1_counts.get(label, 0))
        for label in CLASS_LABELS
    }

    output = {
        "video_path": args.video_path,
        "checkpoint_path": args.checkpoint_path,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed,
        "frame_stride": int(args.frame_stride),
        "diagnose": diagnose,
        "video_level_postprocess": {
            "frame_top1_prompt_counts": frame_top1_prompt_counts,
            "cleft_hard_palate_count": cleft_count,
            "intact_hard_palate_count": intact_count,
            "palate_frame_count": palate_total,
            "palate_cleft_ratio": palate_cleft_ratio,
            "cleft_ratio_all_frames": cleft_ratio_all_frames,
            "palate_cleft_ratio_thr": float(args.palate_cleft_ratio_thr),
        },
        "prompts": DEFAULT_KEYFRAME_PROMPTS,
        "events": events,
    }

    out_json = Path(args.out_json) if args.out_json else Path(args.video_path).with_suffix(".fsm_events.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved event json: {out_json}")
    log(
        f"[post] cleft_hard_palate_count={cleft_count} "
        f"palate_cleft_ratio={palate_cleft_ratio:.4f} "
        f"diagnose={diagnose}"
    )
    log(f"[done] events={len(events)} processed_frames={processed}")


if __name__ == "__main__":
    main()
