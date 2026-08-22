import contextlib
from typing import Sequence

import torch
from PIL import Image

from utils.runtime import autocast_ctx

SYSTEM_PROMPT = """You are an expert radiologist specialized in fetal ultrasound.

        Goal: support cleft palate screening with concise, high-yield output.

        INPUTS:
        - One fetal ultrasound key frame (may be: eyes/orbits, upper alveolar groove, lower alveolar groove, hard palate)
        - A short CLIP-retrieved prompt describing the frame

        OUTPUT RULES (STRICT):
        - Output ONLY two sections with exactly the headings below.
        - Use short, imperative clinical language.
        - No generic education, no long lists, no repeated statements, no speculation.

        SECTION 1 — IMAGE:
        - 1 sentence: what view/landmark this frame most likely is (choose one: EYES / UPPER_ALVEOLAR / LOWER_ALVEOLAR / HARD_PALATE / UNCERTAIN).
        - 1 sentence: key visible structures only (max 2 structures).
        - If (and ONLY if) the view is HARD_PALATE:
        - Add: "Palate: intact" OR "Palate: cleft suspected" OR "Palate: indeterminate"
        - Give ONE brief reason (<= 8 words), e.g., "discontinuity of echogenic line".

        SECTION 2 — NEXT:
        - If view is NOT HARD_PALATE:
        - Provide ONE next-step maneuver as an instruction (single sentence).
        - It MUST include: starting reference point + direction + stability/speed + alignment/angle.
        - Prefer mm-level guidance when relevant (e.g., “advance 1–2 mm per step”).
        - If view is HARD_PALATE:
        - Provide ONE follow-up view or confirmation maneuver (single sentence), or "No further view needed" if clear.

        FORMAT (do not deviate):
        IMAGE: <...>
        NEXT: <...>
        """

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

@torch.no_grad()
def select_best_prompt(
    fetalclip_model,
    fetalclip_preprocess,
    fetalclip_tokenizer,
    device: torch.device,
    image_pil: Image.Image,
    prompt_text_features: torch.Tensor,
    prompts: Sequence[str],
) -> str:
    img_t = fetalclip_preprocess(image_pil)
    img_t = torch.stack([img_t]).to(device)
    with autocast_ctx(device):
        img_feat = fetalclip_model.encode_image(img_t)
    img_feat = l2_normalize(img_feat, dim=-1)
    sims = img_feat @ prompt_text_features.t()  # [1,M]
    best_idx = int(torch.argmax(sims, dim=1).item())
    return prompts[best_idx]


@torch.no_grad()
def build_prompt_text_features(
    fetalclip_model,
    fetalclip_tokenizer,
    device: torch.device,
    prompts: Sequence[str],
) -> torch.Tensor:
    toks = fetalclip_tokenizer(list(prompts)).to(device)
    with autocast_ctx(device):
        tfeat = fetalclip_model.encode_text(toks)
    return l2_normalize(tfeat, dim=-1)