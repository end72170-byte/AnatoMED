"""Incremental video-state utilities for ANATOPAVE inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


SLOT_NAMES: Tuple[str, ...] = (
    "EYES",
    "UPPER_ALVEOLAR",
    "LOWER_ALVEOLAR",
    "HARD_PALATE",
)
SLOT_STATES = ("PENDING", "AVAILABLE", "CLOSED_MISSING")
CLASS_TO_SLOT = {
    "EYES": "EYES",
    "UPPER_ALVEOLAR": "UPPER_ALVEOLAR",
    "LOWER_ALVEOLAR": "LOWER_ALVEOLAR",
    "INTACT_HARD_PALATE": "HARD_PALATE",
    "CLEFT_HARD_PALATE": "HARD_PALATE",
}


def compact_summary_sha256(summary: Optional[torch.Tensor]) -> Optional[str]:
    if summary is None:
        return None
    vector = summary.detach().float().reshape(-1).cpu().contiguous()
    return hashlib.sha256(vector.numpy().tobytes()).hexdigest()


@dataclass
class SlotRecord:
    slot: str
    state: str = "PENDING"
    summary: Optional[torch.Tensor] = None
    label: Optional[str] = None
    prompt: Optional[str] = None
    frame_idx: Optional[int] = None
    timestamp_sec: Optional[float] = None
    clip_score: Optional[float] = None
    clip_margin: Optional[float] = None
    event_index: Optional[int] = None
    event_text: Optional[str] = None
    event_image_path: Optional[str] = None
    event_mask_path: Optional[str] = None
    event_overlay_path: Optional[str] = None
    class_scores: Dict[str, float] = field(default_factory=dict)


class IncrementalSlotCache:
    """Four-slot cache that stores compact summaries, never dense maps.

    A current event is supplied as the main slot to ``composer_inputs``. Other
    available slots are read from this cache. Missing slots remain zero-valued
    placeholders; the Composer turns their availability mask into learned
    unavailable embeddings.
    """

    def __init__(self, summary_dim: int = 768) -> None:
        self.summary_dim = int(summary_dim)
        self.records: Dict[str, SlotRecord] = {
            slot: SlotRecord(slot=slot) for slot in SLOT_NAMES
        }
        self._history: Dict[Tuple[str, str, int], SlotRecord] = {}
        self.representative_strategy = "latest_stable_event"

    def _slot_for_label(self, label: str) -> str:
        try:
            return CLASS_TO_SLOT[str(label)]
        except KeyError as exc:
            raise ValueError(f"Cannot route label to an ANATOPAVE slot: {label}") from exc

    def update(
        self,
        *,
        label: str,
        summary: torch.Tensor,
        frame_idx: int,
        timestamp_sec: float,
        prompt: str,
        clip_score: float,
        clip_margin: float,
        class_scores: Mapping[str, float],
    ) -> str:
        """Replace the representative of a slot after a stable FSM event."""
        slot = self._slot_for_label(label)
        vector = summary.detach().float().reshape(-1).cpu().contiguous()
        if vector.numel() != self.summary_dim:
            raise ValueError(
                f"Expected compact summary dim {self.summary_dim}, got {vector.numel()}"
            )
        record = SlotRecord(
            slot=slot,
            state="AVAILABLE",
            summary=vector,
            label=str(label),
            prompt=str(prompt),
            frame_idx=int(frame_idx),
            timestamp_sec=float(timestamp_sec),
            clip_score=float(clip_score),
            clip_margin=float(clip_margin),
            class_scores={str(key): float(value) for key, value in class_scores.items()},
        )
        self.records[slot] = record
        self._history[(slot, str(label), int(frame_idx))] = record
        return slot

    def attach_event(self, slot: str, event: Mapping[str, Any]) -> None:
        key = (slot, str(event["label"]), int(event["frame_idx"]))
        try:
            record = self._history[key]
        except KeyError as exc:
            raise KeyError(f"No compact evidence snapshot for merged event {key}") from exc
        record.event_index = int(event["event_index"])
        record.event_text = str(event.get("text", ""))
        for field_name in ("image_path", "mask_path", "overlay_path"):
            setattr(record, f"event_{field_name}", event.get(field_name))

    def reconcile_events(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        strategy: str = "highest_clip_score",
    ) -> Dict[str, Optional[Tuple[str, int]]]:
        """Atomically select one final representative snapshot per slot.

        Online updates always use the latest stable event. This finalization
        step may choose a different event after adjacent-event merging, but it
        replaces the full compact evidence snapshot rather than event fields
        alone.
        """
        if strategy not in {"highest_clip_score", "latest_stable_event"}:
            raise ValueError(f"Unsupported slot representative strategy: {strategy}")
        selected: Dict[str, Optional[Tuple[str, int]]] = {}
        for slot in SLOT_NAMES:
            candidates = [event for event in events if event.get("slot") == slot]
            if not candidates:
                selected[slot] = None
                continue
            if strategy == "highest_clip_score":
                chosen = max(
                    candidates,
                    key=lambda event: (
                        float(event.get("clip_score", -1.0)),
                        int(event.get("frame_idx", -1)),
                    ),
                )
            else:
                chosen = max(candidates, key=lambda event: int(event.get("frame_idx", -1)))
            key = (slot, str(chosen["label"]), int(chosen["frame_idx"]))
            try:
                representative = self._history[key]
            except KeyError as exc:
                raise KeyError(f"Merged event has no matching compact evidence snapshot: {key}") from exc
            self.attach_event(slot, chosen)
            self.records[slot] = representative
            selected[slot] = (str(chosen["label"]), int(chosen["frame_idx"]))
        self.representative_strategy = strategy
        return selected

    def close(self) -> None:
        for record in self.records.values():
            if record.state == "PENDING":
                record.state = "CLOSED_MISSING"

    def available(self, slot: str) -> bool:
        return self.records[slot].state == "AVAILABLE"

    def composer_inputs(
        self,
        *,
        main_slot: str,
        main_summary: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``slot_summaries``, ``slot_available``, ``slot_is_main``."""
        if main_slot not in self.records:
            raise ValueError(f"Unknown main slot: {main_slot}")
        current = main_summary.detach().to(device=device, dtype=dtype).reshape(-1)
        if current.numel() != self.summary_dim:
            raise ValueError(f"Main summary has {current.numel()} values, expected {self.summary_dim}")
        summaries = torch.zeros((1, len(SLOT_NAMES), self.summary_dim), device=device, dtype=dtype)
        available = torch.zeros((1, len(SLOT_NAMES)), device=device, dtype=torch.bool)
        main = torch.zeros((1, len(SLOT_NAMES)), device=device, dtype=torch.bool)
        for index, slot in enumerate(SLOT_NAMES):
            if slot == main_slot:
                summaries[0, index] = current
                available[0, index] = True
                main[0, index] = True
            elif self.available(slot):
                value = self.records[slot].summary
                if value is None:
                    raise AssertionError(f"Available slot {slot} has no compact summary")
                summaries[0, index] = value.to(device=device, dtype=dtype)
                available[0, index] = True
        return summaries, available, main

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for slot, record in self.records.items():
            result[slot] = {
                "state": record.state,
                "summary_dim": int(record.summary.numel()) if record.summary is not None else None,
                "summary_sha256": compact_summary_sha256(record.summary),
                "label": record.label,
                "prompt": record.prompt,
                "frame_idx": record.frame_idx,
                "timestamp_sec": record.timestamp_sec,
                "clip_score": record.clip_score,
                "clip_margin": record.clip_margin,
                "event_index": record.event_index,
                "event_text": record.event_text,
                "event_image_path": record.event_image_path,
                "event_mask_path": record.event_mask_path,
                "event_overlay_path": record.event_overlay_path,
                "class_scores": dict(record.class_scores),
            }
        return result


def build_video_diagnosis(
    *,
    events: Sequence[Mapping[str, Any]],
    frame_top1_counts: Mapping[str, int],
    processed_frames: int,
    palate_cleft_ratio_thr: float,
) -> Dict[str, Any]:
    """Build the event-gated normal/abnormal/insufficient-evidence result."""
    has_intact = any(event.get("label") == "INTACT_HARD_PALATE" for event in events)
    has_cleft = any(event.get("label") == "CLEFT_HARD_PALATE" for event in events)
    available = bool(has_intact or has_cleft)
    intact_count = int(frame_top1_counts.get("INTACT_HARD_PALATE", 0))
    cleft_count = int(frame_top1_counts.get("CLEFT_HARD_PALATE", 0))
    palate_count = intact_count + cleft_count
    if not has_intact and not has_cleft:
        event_pattern = "none"
        ratio: Optional[float] = None
        ratio_used = False
        diagnose = "insufficient_evidence"
        reason = "no_stable_hard_palate_event"
    elif has_intact and not has_cleft:
        event_pattern = "intact_only"
        ratio = float(cleft_count / palate_count) if palate_count else 0.0
        ratio_used = False
        diagnose = "normal"
        reason = None
    elif has_cleft and not has_intact:
        event_pattern = "cleft_only"
        ratio = float(cleft_count / palate_count) if palate_count else 1.0
        ratio_used = False
        diagnose = "abnormal"
        reason = None
    else:
        event_pattern = "both"
        ratio = float(cleft_count / palate_count) if palate_count else 0.0
        ratio_used = True
        diagnose = "abnormal" if ratio >= float(palate_cleft_ratio_thr) else "normal"
        reason = None
    return {
        "diagnosis_rule": "stable_event_pattern_then_ratio_v2",
        "diagnosis_event_pattern": event_pattern,
        "palate_cleft_ratio_used_for_diagnosis": ratio_used,
        "hard_palate_available": available,
        "diagnose": diagnose,
        "insufficient_evidence_reason": reason,
        "has_intact_hard_palate_event": bool(has_intact),
        "has_cleft_hard_palate_event": bool(has_cleft),
        "palate_cleft_ratio": ratio,
        "cleft_hard_palate_count": cleft_count,
        "intact_hard_palate_count": intact_count,
        "palate_frame_count": palate_count,
        "cleft_ratio_all_frames": float(cleft_count / processed_frames) if processed_frames else 0.0,
        "palate_cleft_ratio_thr": float(palate_cleft_ratio_thr),
    }


def build_rule_based_summary(
    *,
    slot_cache: IncrementalSlotCache,
    diagnosis: Mapping[str, Any],
) -> Dict[str, Any]:
    snapshot = slot_cache.snapshot()
    available = [slot for slot, item in snapshot.items() if item["state"] == "AVAILABLE"]
    missing = [slot for slot, item in snapshot.items() if item["state"] == "CLOSED_MISSING"]
    representatives = {
        slot: {
            "state": item["state"],
            "label": item["label"],
            "frame_idx": item["frame_idx"],
            "timestamp_sec": item["timestamp_sec"],
            "event_index": item["event_index"],
            "event_text": item["event_text"],
        }
        for slot, item in snapshot.items()
        if item["state"] == "AVAILABLE"
    }
    return {
        "summary_type": "rule_based",
        "slot_representative_strategy": slot_cache.representative_strategy,
        "slot_availability": {slot: item["state"] == "AVAILABLE" for slot, item in snapshot.items()},
        "available_slots": available,
        "missing_slots": missing,
        "representative_events": representatives,
        "hard_palate_available": diagnosis["hard_palate_available"],
        "diagnose": diagnosis["diagnose"],
        "insufficient_evidence_reason": diagnosis["insufficient_evidence_reason"],
        "palate_cleft_ratio": diagnosis["palate_cleft_ratio"],
    }
