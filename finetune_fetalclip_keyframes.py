"""
Finetune FetalCLIP on labeled keyframes to improve keyframe localization/recognition.

Dataset expectation (per patient folder):
- Ultrasound video (unused here), several keyframe images, per-keyframe .txt descriptions,
  a Word doc (ignored in training), and a COCO-style label JSON (ROI coords + categories).
- labels JSON like labels_my-project-name_*.json provides `images` and `annotations` with `categories`.
- Images are resolved under --images_root; text is resolved by matching basename + .txt in the same root.
- Text for training: (a) the per-frame .txt if present; (b) fallback to joined category names for that frame.

Loss: symmetric CLIP InfoNCE over image/text features.

Usage (single GPU):
python finetune_fetalclip_keyframes.py \
  --clip_dataset_root /data/clip-dataset \
  --val_dataset_root /data/clip-dataset \
  --fold_dir /data/splits/fold_0 \
  --out_dir runs/fetalclip

"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.factory import _MODEL_CONFIGS
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils.stage2_dataset import resolve_case_dataset_root, resolve_case_dirs


DEFAULT_KEYFRAME_PROMPTS = [
    "fetal ultrasound, fetal eyes.",
    "fetal ultrasound, upper alveolar ridge.",
    "fetal ultrasound, intact hard palate.",
    "fetal ultrasound, lower alveolar ridge.",
    "fetal ultrasound, cleft hard palate.",
]

PROMPT_TO_ANATOMY = {
    "fetal ultrasound, fetal eyes.": "fetal eyes",
    "fetal ultrasound, upper alveolar ridge.": "upper alveolar ridge",
    "fetal ultrasound, intact hard palate.": "hard palate",
    "fetal ultrasound, lower alveolar ridge.": "lower alveolar ridge",
    "fetal ultrasound, cleft hard palate.": "hard palate",
}

HARD_PALATE_NORMAL_PROMPT = "fetal ultrasound, intact hard palate."
HARD_PALATE_CLEFT_PROMPT = "fetal ultrasound, cleft hard palate."

BEST_METRIC_FIVE_PROMPT = "five_prompt_macro_f1"
BEST_METRIC_BALANCED_ROUTER = "balanced_router"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class KeyframeClipDataset(Dataset):
    def __init__(self, images_root: Path, labels_json: Path, text_ext: str = ".txt") -> None:
        self.images_root = images_root
        self.text_ext = text_ext
        with open(labels_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cat_map = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
        self.id_to_fname = {img["id"]: img["file_name"] for img in data.get("images", [])}
        self.img_to_cats: Dict[str, List[str]] = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            fname = self.id_to_fname.get(img_id)
            if not fname:
                continue
            cat_name = self.cat_map.get(ann.get("category_id"), str(ann.get("category_id")))
            self.img_to_cats.setdefault(fname, []).append(cat_name)
        self.fnames = sorted(self.img_to_cats.keys())

    def __len__(self):
        return len(self.fnames)

    def _load_text(self, fname: str) -> str:
        stem = Path(fname).stem
        txt_path = self.images_root / f"{stem}{self.text_ext}"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()
        cats = self.img_to_cats.get(fname, [])
        if cats:
            return "Ultrasound keyframe showing: " + ", ".join(cats)
        return "Ultrasound keyframe."

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = self.images_root / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image {img_path}")
        image = Image.open(img_path).convert("RGB")
        text = self._load_text(fname)
        return image, text


# -----------------------------------------------------------------------------
# Dataset (clip-dataset layout)
# -----------------------------------------------------------------------------
# clip-dataset structure:
#   <root>/<split>/<case>/<normal|abnormal>/<high|low>/<anatomy_category>/imgs/*.png
#   <root>/<split>/<case>/<normal|abnormal>/<high|low>/<anatomy_category>/<stem>.txt
# txt format (3 lines):
#   line1: clip prompt (brief image description) — used for CLIP finetune
#   line2: detailed description + diagnosis
#   line3: next-step operation suggestion
# For CLIP finetuning only line 1 (clip_prompt) is used as text.

def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClipDatasetKeyframe(Dataset):
    """Dataset for the clip-dataset folder layout (CLIP image-text finetuning).

    Hierarchy:
        <dataset_root>/<split>/<case>/<normal|abnormal>/<high|low>/
            <anatomy_folder>/imgs/<img>.png
            <anatomy_folder>/<img>.txt

    Returns (image_pil, clip_prompt_text) per sample, plus paths for debug.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: Optional[str] = None,
        split_file: Optional[Path] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.split_file = split_file

        case_dirs = resolve_case_dirs(dataset_root, split=split, split_file=split_file)

        self._image_paths: List[Path] = []
        self._txt_paths: List[Path] = []
        self._texts: List[str] = []
        self._scan(case_dirs)

        if not self._image_paths:
            source = split_file if split_file is not None else dataset_root / str(split)
            raise RuntimeError(f"No samples found under {source}")

    def _scan(self, case_dirs: List[Path]) -> None:
        """Walk: case / (normal|abnormal) / (high|low) / anatomy_folder."""
        for case_dir in case_dirs:
            for condition_dir in sorted(case_dir.iterdir()):  # normal / abnormal
                if not condition_dir.is_dir():
                    continue
                for quality_dir in sorted(condition_dir.iterdir()):  # high / low
                    if not quality_dir.is_dir():
                        continue
                    for anatomy_dir in sorted(quality_dir.iterdir()):
                        if not anatomy_dir.is_dir():
                            continue
                        imgs_dir = anatomy_dir / "imgs"
                        if not imgs_dir.is_dir():
                            continue
                        self._scan_anatomy_folder(anatomy_dir, imgs_dir)

    def _scan_anatomy_folder(self, anatomy_dir: Path, imgs_dir: Path) -> None:
        for img_path in sorted(imgs_dir.iterdir()):
            if not img_path.is_file() or not _is_image_file(img_path):
                continue
            stem = img_path.stem

            # txt: <stem>.txt in anatomy_dir
            # txt_path = anatomy_dir / f"{stem}.txt"
            # if not txt_path.exists():
            #     continue

            # 假设 anatomy_dir 是一个 Path 对象
            txt_files = list(anatomy_dir.glob("*.txt"))
            if not txt_files:
                continue  # 如果没有找到任何 txt 文件，则跳过
            txt_path = txt_files[0]  # 取第一个 txt 文件

            lines = txt_path.read_text(encoding="utf-8").splitlines()
            lines = [l.strip() for l in lines if l.strip()]
            if not lines:
                continue

            clip_prompt = lines[0]  # line 1: brief image description for CLIP
            if not clip_prompt:
                continue

            self._image_paths.append(img_path)
            self._txt_paths.append(txt_path)
            self._texts.append(clip_prompt)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self._image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return {
            "image": image,
            "text": self._texts[idx],
            "image_path": str(img_path),
            "txt_path": str(self._txt_paths[idx]),
        }


# -----------------------------------------------------------------------------
# Validation: zero-shot prompt classification
# -----------------------------------------------------------------------------

def _match_gt_label(
    text: str,
    prompts: List[str],
) -> int:
    """Find the index in *prompts* that best matches the sample's ground-truth text.

    Matching strategy (in priority order):
    1. Exact string match.
    2. The prompt that shares the most words with *text*.
    Returns -1 if no overlap is found at all.
    """
    text_lower = text.lower().strip()
    for i, p in enumerate(prompts):
        if text_lower == p.lower().strip():
            return i
    # Word-overlap fallback
    text_words = set(text_lower.split())
    best_idx, best_overlap = -1, 0
    for i, p in enumerate(prompts):
        overlap = len(text_words & set(p.lower().split()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    return best_idx


def _safe_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def _compute_multiclass_metrics(
    preds: List[int],
    gts: List[int],
    class_names: List[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    n = len(preds)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "n_samples": 0,
            "per_class": {},
        }

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    correct = 0
    for pred, gt in zip(preds, gts):
        if pred == gt:
            tp[gt] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for i, name in enumerate(class_names):
        metrics = _safe_prf(tp[i], fp[i], fn[i])
        per_class[name] = {
            **metrics,
            "tp": tp[i],
            "fp": fp[i],
            "fn": fn[i],
        }
        if (tp[i] + fn[i]) > 0:
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])

    return {
        "accuracy": correct / n,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "n_samples": n,
        "per_class": per_class,
    }


def _compute_hard_palate_metrics(
    preds: List[int],
    gts: List[int],
    prompts: List[str],
) -> Dict[str, Any]:
    hard_palate_indices = [
        i for i, p in enumerate(prompts)
        if PROMPT_TO_ANATOMY.get(p, p) == "hard palate"
    ]
    hard_palate_index_set = set(hard_palate_indices)
    normal_idx = prompts.index(HARD_PALATE_NORMAL_PROMPT)
    cleft_idx = prompts.index(HARD_PALATE_CLEFT_PROMPT)

    gt_hard_palate_pairs = [
        (pred, gt)
        for pred, gt in zip(preds, gts)
        if gt in hard_palate_index_set
    ]
    routed_pairs = [
        (pred, gt)
        for pred, gt in gt_hard_palate_pairs
        if pred in hard_palate_index_set
    ]
    n_gt_hard_palate = len(gt_hard_palate_pairs)
    n_routed_hard_palate = len(routed_pairs)
    n_route_missed = n_gt_hard_palate - n_routed_hard_palate

    routing = {
        "n_gt_hard_palate": n_gt_hard_palate,
        "n_routed_hard_palate": n_routed_hard_palate,
        "n_route_missed": n_route_missed,
        "hard_palate_routing_recall": (
            n_routed_hard_palate / n_gt_hard_palate if n_gt_hard_palate else 0.0
        ),
    }

    if not routed_pairs:
        conditional_binary = {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "n_samples": 0,
            "normal": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
            "abnormal": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0},
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }
    else:
        tn = fp = fn = tp = 0
        for pred, gt in routed_pairs:
            if gt == cleft_idx and pred == cleft_idx:
                tp += 1
            elif gt == cleft_idx and pred == normal_idx:
                fn += 1
            elif gt == normal_idx and pred == cleft_idx:
                fp += 1
            elif gt == normal_idx and pred == normal_idx:
                tn += 1
            else:
                raise AssertionError("A routed hard-palate pair has an unexpected class index")

        abnormal_metrics = _safe_prf(tp, fp, fn)
        normal_metrics = _safe_prf(tn, fn, fp)
        conditional_binary = {
            "accuracy": (tp + tn) / len(routed_pairs),
            "macro_precision": 0.5 * (normal_metrics["precision"] + abnormal_metrics["precision"]),
            "macro_recall": 0.5 * (normal_metrics["recall"] + abnormal_metrics["recall"]),
            "macro_f1": 0.5 * (normal_metrics["f1"] + abnormal_metrics["f1"]),
            "n_samples": len(routed_pairs),
            "normal": {
                **normal_metrics,
                "tp": tn,
                "fp": fn,
                "fn": fp,
            },
            "abnormal": {
                **abnormal_metrics,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    correct_state = sum(pred == gt for pred, gt in routed_pairs)
    wrong_state = n_routed_hard_palate - correct_state
    end_to_end = {
        "n_gt_hard_palate": n_gt_hard_palate,
        "correct_state": correct_state,
        "wrong_state": wrong_state,
        "unresolved_due_to_route_miss": n_route_missed,
        "resolved_correct_rate": (
            correct_state / n_gt_hard_palate if n_gt_hard_palate else 0.0
        ),
    }
    return {
        "routing": routing,
        "conditional_binary": conditional_binary,
        "end_to_end": end_to_end,
    }


def _harmonic_mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def compute_best_selection_score(metrics: Dict[str, Any], best_metric: str) -> float:
    """Return the checkpoint score without changing the CLIP training objective."""
    if best_metric == BEST_METRIC_FIVE_PROMPT:
        return float(metrics["macro_f1"])
    if best_metric == BEST_METRIC_BALANCED_ROUTER:
        return _harmonic_mean([
            metrics["anatomy_macro_f1"],
            metrics["hard_palate_binary"]["macro_f1"],
        ])
    raise ValueError(f"Unknown best metric: {best_metric}")


def _write_json(path: Path, value: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


@torch.no_grad()
def validate_prompt_classification(
    model: torch.nn.Module,
    preprocess: Any,
    tokenizer: Any,
    val_loader: DataLoader,
    device: torch.device,
    prompts: List[str],
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Validate by zero-shot prompt classification.

    For each image, compute cosine similarity against all *prompts* text
    features and predict the class with highest similarity.  Ground-truth
    class is derived from the sample's own text via word-overlap matching.

    Returns dict with per-class precision/recall/f1, macro averages, and accuracy.
    """
    model.eval()

    # Pre-encode all prompt text features
    prompt_toks = tokenizer(prompts).to(device)
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        prompt_feats = model.encode_text(prompt_toks)
    prompt_feats = F.normalize(prompt_feats, dim=-1)  # [K, D]

    all_preds: List[int] = []
    all_gts: List[int] = []
    skipped = 0

    for batch in val_loader:
        images = batch["images"]
        texts = batch["texts"]

        imgs_t = torch.stack([preprocess(im) for im in images]).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            img_feats = model.encode_image(imgs_t)
        img_feats = F.normalize(img_feats, dim=-1)  # [B, D]

        sims = img_feats @ prompt_feats.t()  # [B, K]
        preds = sims.argmax(dim=1).cpu().tolist()

        for pred, txt in zip(preds, texts):
            gt = _match_gt_label(txt, prompts)
            if gt < 0:
                skipped += 1
                continue
            all_preds.append(pred)
            all_gts.append(gt)

    model.train()

    prompt_metrics = _compute_multiclass_metrics(all_preds, all_gts, prompts)

    anatomy_names = [
        "fetal eyes",
        "upper alveolar ridge",
        "hard palate",
        "lower alveolar ridge",
    ]
    anatomy_to_idx = {name: i for i, name in enumerate(anatomy_names)}
    anatomy_preds = [anatomy_to_idx[PROMPT_TO_ANATOMY.get(prompts[p], prompts[p])] for p in all_preds]
    anatomy_gts = [anatomy_to_idx[PROMPT_TO_ANATOMY.get(prompts[g], prompts[g])] for g in all_gts]
    anatomy_metrics = _compute_multiclass_metrics(anatomy_preds, anatomy_gts, anatomy_names)

    hard_palate_metrics = _compute_hard_palate_metrics(all_preds, all_gts, prompts)

    return {
        **prompt_metrics,
        "n_skipped": skipped,
        "per_anatomy": anatomy_metrics["per_class"],
        "anatomy_accuracy": anatomy_metrics["accuracy"],
        "anatomy_macro_f1": anatomy_metrics["macro_f1"],
        "anatomy_macro_precision": anatomy_metrics["macro_precision"],
        "anatomy_macro_recall": anatomy_metrics["macro_recall"],
        "hard_palate_routing": hard_palate_metrics["routing"],
        "hard_palate_binary": hard_palate_metrics["conditional_binary"],
        "hard_palate_end_to_end": hard_palate_metrics["end_to_end"],
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def clip_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, logit_scale: torch.Tensor):
    # image_feats/text_feats already normalized; logit_scale is scalar
    logits_per_image = logit_scale * image_feats @ text_feats.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(image_feats.size(0), device=image_feats.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


def _collate_clip(batch):
    """Collate function supporting both old (tuple) and new (dict) dataset formats."""
    if isinstance(batch[0], dict):
        return {
            "images": [b["image"] for b in batch],
            "texts": [b["text"] for b in batch],
            "image_paths": [b.get("image_path", "") for b in batch],
            "txt_paths": [b.get("txt_path", "") for b in batch],
        }
    # Compatibility tuple format: (image, text)
    images, texts = zip(*batch)
    return {"images": list(images), "texts": list(texts), "image_paths": [], "txt_paths": []}


def _prune_checkpoints(out_dir: Path, max_checkpoints: int) -> None:
    """Keep at most *max_checkpoints* total weight files in out_dir.

    Priority:
    1. Always keep fetalclip_finetune_best.pt when it exists.
    2. Keep the newest epoch checkpoints.
    """
    best_path = out_dir / "fetalclip_finetune_best.pt"
    epoch_ckpts: List[Tuple[int, Path]] = []
    pattern = re.compile(r"fetalclip_finetune_epoch(\d+)\.pt$")

    for path in out_dir.glob("fetalclip_finetune_epoch*.pt"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        epoch_ckpts.append((int(match.group(1)), path))

    epoch_ckpts.sort(key=lambda item: item[0], reverse=True)

    reserved_slots = 1 if best_path.exists() else 0
    keep_epoch_count = max(max_checkpoints - reserved_slots, 0)
    for _, ckpt_path in epoch_ckpts[keep_epoch_count:]:
        ckpt_path.unlink(missing_ok=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"

    if args.max_checkpoints < 1:
        raise ValueError("--max_checkpoints must be >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config & model
    with open(args.fetalclip_config, "r") as f:
        cfg = json.load(f)
    _MODEL_CONFIGS["FetalCLIP"] = cfg
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=args.fetalclip_weights
    )
    model.to(device)
    model.train()

    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    # Dataset selection
    fold_dir = Path(args.fold_dir) if args.fold_dir else None
    train_split_file = fold_dir / "train.txt" if fold_dir is not None else None
    val_split_file = fold_dir / "val.txt" if fold_dir is not None else None
    val_dataset_root = Path(args.val_dataset_root) if args.val_dataset_root else Path(args.clip_dataset_root)

    if args.clip_dataset_root and (train_split_file is not None or args.clip_dataset_split):
        if train_split_file is not None:
            dataset = ClipDatasetKeyframe(
                dataset_root=resolve_case_dataset_root(Path(args.clip_dataset_root)),
                split_file=train_split_file,
            )
            print(f"[data] Using ClipDatasetKeyframe from fold train file: {len(dataset)} samples")
        else:
            dataset = ClipDatasetKeyframe(
                dataset_root=Path(args.clip_dataset_root),
                split=args.clip_dataset_split,
            )
            print(f"[data] Using ClipDatasetKeyframe: {len(dataset)} samples")
    else:
        dataset = KeyframeClipDataset(Path(args.images_root), Path(args.labels_json), text_ext=args.text_ext)
        print(f"[data] Using KeyframeClipDataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate_clip,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)

    # Validation dataset
    val_loader: Optional[DataLoader] = None
    if val_split_file is not None:
        try:
            val_ds = ClipDatasetKeyframe(
                dataset_root=resolve_case_dataset_root(val_dataset_root),
                split_file=val_split_file,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=_collate_clip,
            )
            print(f"[data] Val fold file '{val_split_file}': {len(val_ds)} samples")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[warn] Cannot load val split file '{val_split_file}': {e}; skipping validation.")
    elif args.val_split:
        try:
            val_ds = ClipDatasetKeyframe(
                dataset_root=val_dataset_root,
                split=args.val_split,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=_collate_clip,
            )
            print(f"[data] Val split '{args.val_split}': {len(val_ds)} samples")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[warn] Cannot load val split '{args.val_split}': {e}; skipping validation.")

    # Early stopping state
    best_selection_score = -1.0
    patience_counter = 0
    patience = int(args.patience)
    validation_history: List[Dict[str, Any]] = []
    history_path = Path(args.out_dir) / "validation_history.json"

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(loader):
            imgs = batch["images"]
            texts = batch["texts"]
            optim.zero_grad()
            imgs_proc = torch.stack([preprocess_train(im) for im in imgs]).to(device)
            toks = tokenizer(list(texts)).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                image_feats = model.encode_image(imgs_proc)
                text_feats = model.encode_text(toks)
                image_feats = F.normalize(image_feats, dim=-1)
                text_feats = F.normalize(text_feats, dim=-1)
                loss = clip_loss(image_feats, text_feats, model.logit_scale.exp())
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item() * imgs_proc.size(0)
            count += imgs_proc.size(0)
            if (step + 1) % args.log_every == 0:
                print(f"Epoch {epoch+1} Step {step+1}/{len(loader)} Loss {loss.item():.4f}")

        avg = total / max(count, 1)
        print(f"Epoch {epoch+1} done. Avg loss {avg:.4f}")

        # ---- Validation ----
        if val_loader is not None:
            val_metrics = validate_prompt_classification(
                model, preprocess_val, tokenizer, val_loader,
                device, DEFAULT_KEYFRAME_PROMPTS, use_amp=use_amp,
            )
            print(
                f"  [Val][5-prompt] acc={val_metrics['accuracy']:.4f}  "
                f"macro_f1={val_metrics['macro_f1']:.4f}  "
                f"macro_prec={val_metrics['macro_precision']:.4f}  "
                f"macro_rec={val_metrics['macro_recall']:.4f}  "
                f"(n={val_metrics['n_samples']}, skipped={val_metrics['n_skipped']})"
            )
            for pname, pmet in val_metrics["per_class"].items():
                print(
                    f"    {pname}: "
                    f"P={pmet['precision']:.3f} R={pmet['recall']:.3f} F1={pmet['f1']:.3f} "
                    f"(TP={pmet['tp']} FP={pmet['fp']} FN={pmet['fn']})"
                )

            print(
                f"  [Val][4-anatomy] acc={val_metrics['anatomy_accuracy']:.4f}  "
                f"macro_f1={val_metrics['anatomy_macro_f1']:.4f}  "
                f"macro_prec={val_metrics['anatomy_macro_precision']:.4f}  "
                f"macro_rec={val_metrics['anatomy_macro_recall']:.4f}"
            )
            for aname, amet in val_metrics["per_anatomy"].items():
                print(
                    f"    {aname}: "
                    f"P={amet['precision']:.3f} R={amet['recall']:.3f} F1={amet['f1']:.3f} "
                    f"(TP={amet['tp']} FP={amet['fp']} FN={amet['fn']})"
                )

            hp_routing = val_metrics["hard_palate_routing"]
            print(
                f"  [Val][hard-palate-routing] "
                f"recall={hp_routing['hard_palate_routing_recall']:.4f}  "
                f"routed={hp_routing['n_routed_hard_palate']}/"
                f"{hp_routing['n_gt_hard_palate']}  "
                f"missed={hp_routing['n_route_missed']}"
            )

            hp = val_metrics["hard_palate_binary"]
            print(
                f"  [Val][hard-palate-binary|routed] acc={hp['accuracy']:.4f}  "
                f"macro_f1={hp['macro_f1']:.4f}  "
                f"normal_f1={hp['normal']['f1']:.4f}  "
                f"abnormal_f1={hp['abnormal']['f1']:.4f}  "
                f"(n={hp['n_samples']})"
            )
            print(
                f"    normal: "
                f"P={hp['normal']['precision']:.3f} R={hp['normal']['recall']:.3f} F1={hp['normal']['f1']:.3f} "
                f"(TP={hp['normal']['tp']} FP={hp['normal']['fp']} FN={hp['normal']['fn']})"
            )
            print(
                f"    abnormal: "
                f"P={hp['abnormal']['precision']:.3f} R={hp['abnormal']['recall']:.3f} F1={hp['abnormal']['f1']:.3f} "
                f"(TP={hp['abnormal']['tp']} FP={hp['abnormal']['fp']} FN={hp['abnormal']['fn']})"
            )
            print(
                f"    confusion: TN={hp['confusion']['tn']} FP={hp['confusion']['fp']} "
                f"FN={hp['confusion']['fn']} TP={hp['confusion']['tp']}"
            )
            hp_e2e = val_metrics["hard_palate_end_to_end"]
            print(
                f"  [Val][hard-palate-end-to-end] "
                f"resolved_correct_rate={hp_e2e['resolved_correct_rate']:.4f}  "
                f"correct={hp_e2e['correct_state']}  "
                f"wrong={hp_e2e['wrong_state']}  "
                f"unresolved={hp_e2e['unresolved_due_to_route_miss']}  "
                f"(n={hp_e2e['n_gt_hard_palate']})"
            )

            current_selection_score = compute_best_selection_score(val_metrics, args.best_metric)
            validation_history.append({
                "epoch": epoch + 1,
                "train_loss": avg,
                "best_metric": args.best_metric,
                "selection_score": current_selection_score,
                "metrics": val_metrics,
            })
            _write_json(history_path, validation_history)
            print(
                f"  [Val][selection] metric={args.best_metric} "
                f"score={current_selection_score:.4f}"
            )
            if current_selection_score > best_selection_score:
                best_selection_score = current_selection_score
                patience_counter = 0
                # Save best checkpoint
                best_ckpt = {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "best_metric": args.best_metric,
                    "selection_score": current_selection_score,
                    "args": vars(args),
                }
                torch.save(best_ckpt, Path(args.out_dir) / "fetalclip_finetune_best.pt")
                print(
                    f"  [Val] New best {args.best_metric}={best_selection_score:.4f}, "
                    "saved best checkpoint."
                )
            else:
                patience_counter += 1
                if patience > 0:
                    print(f"  [Val] No improvement ({patience_counter}/{patience}).")
                else:
                    print("  [Val] No improvement (early stopping disabled).")
                if patience > 0 and patience_counter >= patience:
                    print(f"  [Val] Early stopping triggered after {patience} epochs without improvement.")
                    break

        # Periodic checkpoint
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
        }
        torch.save(ckpt, Path(args.out_dir) / f"fetalclip_finetune_epoch{epoch+1}.pt")
        _prune_checkpoints(Path(args.out_dir), args.max_checkpoints)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Finetune FetalCLIP on keyframe image-text pairs")
    # Old dataset mode (KeyframeClipDataset)
    p.add_argument("--images_root", type=str, default="", help="Directory containing keyframe images and .txt descriptions")
    p.add_argument("--labels_json", type=str, default="", help="COCO-style JSON with image_id/file_name/category mapping")
    p.add_argument("--text_ext", type=str, default=".txt", help="Per-frame description extension")
    # New dataset mode (ClipDatasetKeyframe)
    p.add_argument("--clip_dataset_root", type=str, required=True, help="Clip dataset root. For fold-based CV, point this to the case root such as .../all.")
    p.add_argument("--val_dataset_root", type=str, required=True, help="Validation dataset root.")
    p.add_argument("--clip_dataset_split", type=str, default="train", help="Split folder name under clip_dataset_root when not using --fold_dir")
    p.add_argument("--val_split", type=str, default="val", help="Validation split folder name under clip_dataset_root when not using --fold_dir. Empty string disables validation.")
    p.add_argument("--fold_dir", type=str, default="", help="Fold directory containing train.txt and val.txt. When provided, it overrides --clip_dataset_split/--val_split.")
    # Model
    p.add_argument("--fetalclip_config", type=str, default="model/FetalCLIP_config.json")
    p.add_argument("--fetalclip_weights", type=str, default="model/FetalCLIP_weights.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="runs/fetalclip")
    p.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoint files to keep in out_dir, including the best checkpoint.")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 = disabled).")
    p.add_argument(
        "--best_metric",
        choices=[BEST_METRIC_FIVE_PROMPT, BEST_METRIC_BALANCED_ROUTER],
        default=BEST_METRIC_FIVE_PROMPT,
        help=(
            "Checkpoint selection metric. balanced_router is the harmonic mean of "
            "four-anatomy macro F1 and conditional hard-palate binary macro F1."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
