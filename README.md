# ANATOPAVE

This repository contains the source code for the ANATOPAVE fetal-ultrasound
pipeline used by the paper: FetalCLIP prompt routing, temporal FSM aggregation,
anatomy-aware evidence composition, shared grounding, and segmentation.

The repository intentionally excludes patient data, image/video manifests,
checkpoints, generated metrics, and experiment logs. Supply those inputs at
runtime with command-line arguments. Local outputs should be written below
`runs/`, which is ignored by Git.

## Main entry points

- `finetune_fetalclip_keyframes.py`: train the prompt classifier.
- `eval_fetalclip_baseline.py`: evaluate the frozen FetalCLIP baseline.
- `train_seg_fetalclip_unetr.py`: train the FetalCLIP-conditioned segmentation decoder.
- `train_anatopave.py`: train the joint Composer and shared-grounding model.
- `eval_anatopave.py`: evaluate a joint checkpoint.
- `inference_anatopave.py`: run incremental video inference and FSM aggregation.
- `eval_video_oof_fsm.py`: run fold-specific video-level OOF FSM inference.

Dataset manifests, fold splits, model checkpoints, and output paths are required
inputs or explicit command-line options; no machine-specific absolute paths are
embedded in the active entry points.

The case-grouped baseline split is versioned under
`splits/sam-dataset/group_kfold_k5_seed42_anatopave/`. A controlled real-data
demo is provided under `demo-dataset/`; read its README before use.

## Verification

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
