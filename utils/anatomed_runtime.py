"""Shared model/data runtime for AnatoMED training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import open_clip
import torch
from open_clip.factory import _MODEL_CONFIGS

from models.anatomed import AnatoMEDHeads
from train_seg_fetalclip_unetr import _extract_state_dict
from utils.anatomed_cache import CompactSummaryCache, assemble_slot_summaries
from utils.evidence_injection import EvidenceEmbeddingInjector
from utils.models import FetalCLIPSegFeatureExtractor, extract_seg_token_feature
from utils.prompt_utils import SYSTEM_PROMPT
from utils.runtime import autocast_ctx
from utils.vlm_inputs import (
    _get_pad_token_id,
    build_anatomed_generation_inputs,
    build_anatomed_teacher_forcing_inputs,
    pad_and_stack_vlm_inputs,
)


SPECIAL_TOKENS = ("<seg>", "<evidence>")


def load_frozen_fetalclip(
    config_path: Path, weights_path: Path, device: torch.device
) -> Tuple[Any, Any, FetalCLIPSegFeatureExtractor]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _MODEL_CONFIGS["FetalCLIP"] = config
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=None)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    extractor = FetalCLIPSegFeatureExtractor(model.visual)
    extractor.eval()
    return model, preprocess, extractor


def load_vlm_and_processor(
    model_id: str,
    device: torch.device,
    *,
    local_files_only: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Tuple[Any, Any, int, int, int]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        vlm = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, local_files_only=local_files_only
        ).to(device)
    except TypeError:
        vlm = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=local_files_only
        ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=local_files_only
    )
    processor.tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    vlm.resize_token_embeddings(len(processor.tokenizer))
    seg_token_id = int(processor.tokenizer.convert_tokens_to_ids("<seg>"))
    evidence_token_id = int(processor.tokenizer.convert_tokens_to_ids("<evidence>"))
    if seg_token_id == evidence_token_id or min(seg_token_id, evidence_token_id) < 0:
        raise RuntimeError("Failed to register distinct <seg> and <evidence> tokens")

    if lora_r > 0:
        from peft import LoraConfig, get_peft_model

        vlm = get_peft_model(
            vlm,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    else:
        for parameter in vlm.parameters():
            parameter.requires_grad = False
    vlm_dim = int(vlm.config.text_config.hidden_size)
    return vlm, processor, seg_token_id, evidence_token_id, vlm_dim


def build_heads(
    *,
    summary_dim: int,
    dense_feature_dim: int,
    vlm_dim: int,
    composer_dim: int,
    grounding_dim: int,
    num_heads: int,
    seg_pretrain_path: Path,
    device: torch.device,
) -> AnatoMEDHeads:
    heads = AnatoMEDHeads(
        summary_dim=summary_dim,
        dense_feature_dim=dense_feature_dim,
        vlm_dim=vlm_dim,
        composer_dim=composer_dim,
        grounding_dim=grounding_dim,
        num_heads=num_heads,
    )
    checkpoint = torch.load(seg_pretrain_path, map_location="cpu", weights_only=False)
    heads.decoder.load_segmentation_pretrain(checkpoint)
    return heads.to(device)


def prepare_language_batch(
    *,
    processor: Any,
    raw_images: Sequence[Any],
    prompts: Sequence[str],
    target_texts: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    inputs, labels = build_anatomed_teacher_forcing_inputs(
        processor, list(raw_images), list(prompts), list(target_texts), SYSTEM_PROMPT
    )
    kwargs = pad_and_stack_vlm_inputs(
        inputs, device, _get_pad_token_id(processor), labels
    )
    return kwargs


def forward_joint_batch(
    *,
    batch: Mapping[str, Any],
    device: torch.device,
    feature_extractor: FetalCLIPSegFeatureExtractor,
    summary_cache: CompactSummaryCache,
    heads: AnatoMEDHeads,
    vlm: Any,
    processor: Any,
    injector: EvidenceEmbeddingInjector,
    seg_token_id: int,
    segmentation_criterion: Any,
    lambda_lm: float,
    lambda_seg: float,
    use_composer: bool = True,
) -> Dict[str, Any]:
    main_images = batch["main_image"].to(device, non_blocking=True)
    main_masks = batch["main_mask"].to(device, non_blocking=True)
    slot_available = batch["slot_available"].to(device)
    slot_is_main = batch["slot_is_main"].to(device)
    with torch.no_grad(), autocast_ctx(device):
        dense_features, main_summary = feature_extractor(main_images)
    slot_summaries = assemble_slot_summaries(
        main_summary=main_summary,
        slot_available=slot_available,
        slot_is_main=slot_is_main,
        slot_image_paths=batch["slot_image_paths"],
        cache=summary_cache,
    )
    with autocast_ctx(device):
        composed = heads.compose(slot_summaries, slot_available, slot_is_main)
    language_kwargs = prepare_language_batch(
        processor=processor,
        raw_images=batch["main_image_raw"],
        prompts=batch["main_clip_prompt"],
        target_texts=batch["main_text"],
        device=device,
    )
    language_kwargs["labels"] = language_kwargs["labels"].masked_fill(
        language_kwargs["input_ids"].eq(injector.evidence_token_id), -100
    )
    with autocast_ctx(device):
        vlm_output = injector.forward(
            vlm, composed["language_embedding"][:, 0],
            {**language_kwargs, "output_hidden_states": True, "return_dict": True},
        )
        loss_lm = vlm_output.loss
        seg_hidden = extract_seg_token_feature(
            vlm_output.hidden_states[-1], language_kwargs["input_ids"], seg_token_id
        )
        main_slot_index = slot_is_main.to(torch.long).argmax(dim=1)
        seg_logits = heads.segment(
            features=dense_features,
            seg_hidden=seg_hidden,
            segmentation_context=composed["segmentation_context"],
            main_slot_index=main_slot_index,
            use_composer=use_composer,
        )
        seg_parts = segmentation_criterion(seg_logits, main_masks)
        loss_seg = seg_parts["total"]
        total = float(lambda_lm) * loss_lm + float(lambda_seg) * loss_seg
    return {
        "loss": total,
        "loss_lm": loss_lm,
        "loss_seg": loss_seg,
        "seg_parts": seg_parts,
        "seg_logits": seg_logits,
        "main_masks": main_masks,
        "composed": composed,
        "dense_features": dense_features,
        "seg_hidden": seg_hidden,
        "language_kwargs": language_kwargs,
    }


@torch.no_grad()
def generate_one(
    *,
    batch: Mapping[str, Any],
    processor: Any,
    vlm: Any,
    injector: EvidenceEmbeddingInjector,
    language_embedding: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 8,
) -> torch.Tensor:
    inputs = build_anatomed_generation_inputs(
        processor, [batch["main_image_raw"][0]], [batch["main_clip_prompt"][0]], SYSTEM_PROMPT
    )
    kwargs = pad_and_stack_vlm_inputs(inputs, device, _get_pad_token_id(processor))
    prompt_length = int(kwargs["input_ids"].shape[1])
    generated = injector.generate(
        vlm, language_embedding[:1, 0], kwargs,
        max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
    )
    # Decoder-only generate() returns prompt + continuation. Metrics must only
    # compare the assistant continuation with the reference report.
    return generated[:, prompt_length:]
