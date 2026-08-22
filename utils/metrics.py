from collections import Counter
from math import exp, log
from typing import List, Sequence

import numpy as np
import torch


def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Dice score from raw logits.  Returns mean over the batch."""
    probs = torch.sigmoid(logits).flatten(1)
    preds = (probs >= threshold).float()
    targets = targets.flatten(1).float()
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()

def compute_ap_score(logits: torch.Tensor, targets: torch.Tensor, iou_threshold: float = 0.5) -> float:
    """Detection-level Average Precision proxy (Mask Bounding Box AP).
    Evaluates whether the predicted region overlaps with the GT region > iou_threshold.
    """
    probs = torch.sigmoid(logits).detach().cpu()
    targets_cpu = targets.detach().cpu()
    B = probs.shape[0]
    
    aps: List[float] = []
    for i in range(B):
        p_mask = (probs[i, 0].numpy() > 0.5).astype(np.uint8)
        t_mask = targets_cpu[i, 0].numpy().astype(np.uint8)
        
        # 1. 如果没有GT，预测有则得0分，无则得满分
        if t_mask.sum() == 0:
            aps.append(1.0 if p_mask.sum() == 0 else 0.0)
            continue
            
        # 2. 如果有GT但没预测出来，得0分
        if p_mask.sum() == 0:
            aps.append(0.0)
            continue
            
        # 3. 提取外接矩形 (Bounding Box)
        # 用 bbox 来计算可以极大降低“粗细不一”带来的影响
        y_true, x_true = np.where(t_mask > 0)
        box_true = [x_true.min(), y_true.min(), x_true.max(), y_true.max()]
        
        y_pred, x_pred = np.where(p_mask > 0)
        box_pred = [x_pred.min(), y_pred.min(), x_pred.max(), y_pred.max()]
        
        # 4. 计算 IoU
        xA = max(box_true[0], box_pred[0])
        yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2])
        yB = min(box_true[3], box_pred[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box_true_area = (box_true[2] - box_true[0] + 1) * (box_true[3] - box_true[1] + 1)
        box_pred_area = (box_pred[2] - box_pred[0] + 1) * (box_pred[3] - box_pred[1] + 1)
        
        iou = interArea / float(box_true_area + box_pred_area - interArea)
        
        # 5. 根据阈值判断 Hit
        aps.append(1.0 if iou >= iou_threshold else 0.0)
        
    return float(np.mean(aps))


def compute_ap_scorem(logits: torch.Tensor, targets: torch.Tensor, iou_threshold: float = 0.5) -> float:
    """Detection-style AP proxy using mask IoU instead of bounding-box IoU."""
    probs = torch.sigmoid(logits).detach().cpu()
    targets_cpu = targets.detach().cpu()
    B = probs.shape[0]

    aps: List[float] = []
    for i in range(B):
        p_mask = (probs[i, 0].numpy() > 0.5).astype(np.uint8)
        t_mask = targets_cpu[i, 0].numpy().astype(np.uint8)

        if t_mask.sum() == 0:
            aps.append(1.0 if p_mask.sum() == 0 else 0.0)
            continue

        if p_mask.sum() == 0:
            aps.append(0.0)
            continue

        inter = np.logical_and(p_mask > 0, t_mask > 0).sum(dtype=np.float64)
        union = np.logical_or(p_mask > 0, t_mask > 0).sum(dtype=np.float64)
        iou = 0.0 if union <= 0.0 else float(inter / union)
        aps.append(1.0 if iou >= iou_threshold else 0.0)

    return float(np.mean(aps))

def compute_ap_score_pixelwise(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Pixel-level Average Precision from raw logits. Returns mean AP over the batch.
    differs from detection-level AP."""
    probs = torch.sigmoid(logits).detach().cpu()
    targets_cpu = targets.detach().cpu()
    B = probs.shape[0]
    aps: List[float] = []
    for i in range(B):
        p = probs[i].flatten().numpy().astype(np.float64)
        t = targets_cpu[i].flatten().numpy().astype(np.int32)
        n_pos = int(t.sum())
        if n_pos == 0:
            aps.append(1.0 if int((p > 0.5).sum()) == 0 else 0.0)
            continue
        sorted_idx = np.argsort(-p)
        sorted_t = t[sorted_idx]
        tp_cumsum = np.cumsum(sorted_t)
        precision = tp_cumsum / np.arange(1, len(sorted_t) + 1, dtype=np.float64)
        ap = float(np.sum(precision * sorted_t) / n_pos)
        aps.append(ap)
    return float(np.mean(aps))

def compute_bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    device: str = "cpu",
    batch_size: int = 8,
    model_type: str = "microsoft/deberta-large-mnli",
) -> float:
    """Compute mean BERTScore-F1 for prediction/reference text pairs."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0

    try:
        from bert_score import score as bertscore_score
    except Exception as e:
        raise RuntimeError(
            "BERTScore requires the 'bert-score' package. Install it with: pip install bert-score"
        ) from e

    _, _, f1 = bertscore_score(
        cands=list(predictions),
        refs=list(references),
        lang="en",
        model_type=model_type,
        batch_size=int(batch_size),
        device=device,
        verbose=False,
    )
    return float(f1.mean().item())


def _extract_ngrams(tokens: Sequence[str], order: int) -> Counter:
    if order <= 0 or len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def compute_bleu_score(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    max_order: int = 4,
    smooth: bool = True,
    lowercase: bool = True,
) -> float:
    """Compute corpus BLEU score for prediction/reference text pairs.

    Returns a value in [0, 1]. Uses simple additive smoothing by default so the
    score is less brittle on short validation/test sets.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0
    if max_order < 1:
        raise ValueError("max_order must be >= 1")

    matches_by_order = [0.0] * max_order
    possible_matches_by_order = [0.0] * max_order
    pred_length = 0
    ref_length = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        if lowercase:
            pred_tokens = [tok.lower() for tok in pred_tokens]
            ref_tokens = [tok.lower() for tok in ref_tokens]

        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngrams = _extract_ngrams(pred_tokens, order)
            ref_ngrams = _extract_ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += float(sum(overlap.values()))
            possible_matches = max(len(pred_tokens) - order + 1, 0)
            possible_matches_by_order[order - 1] += float(possible_matches)

    precisions: List[float] = []
    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        if smooth:
            precisions.append((matches + 1.0) / (possible + 1.0))
        elif possible > 0:
            precisions.append(matches / possible)
        else:
            precisions.append(0.0)

    if min(precisions) <= 0.0:
        return 0.0

    geo_mean = exp(sum(log(p) for p in precisions) / max_order)
    if pred_length == 0:
        return 0.0
    if pred_length > ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1.0 - float(ref_length) / float(pred_length))

    return float(brevity_penalty * geo_mean)
