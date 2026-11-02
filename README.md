# AnatoMED

Method name: **AnatoMED**. Dataset name: **CP-SET**.

Code: https://github.com/end72170-byte/AnatoMED

This repository contains the source code for the AnatoMED fetal-ultrasound
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
- `train_anatomed.py`: train the joint Composer and shared-grounding model.
- `eval_anatomed.py`: evaluate a joint checkpoint.
- `inference_anatomed.py`: run incremental video inference and FSM aggregation.
- `eval_video_oof_fsm.py`: run fold-specific video-level OOF FSM inference.

Dataset manifests, fold splits, model checkpoints, and output paths are required
inputs or explicit command-line options; no machine-specific absolute paths are
embedded in the active entry points.

The case-grouped baseline split is versioned under
`splits/sam-dataset/group_kfold_k5_seed42_anatomed/`. A controlled real-data
demo is provided under `demo-dataset/`; read its README before use.

## Full historical search core

`revision_batch1/search.py` provides the `AnatomicalSearch` module used by the
Full historical-search experiments: causal four-role memory, candidate-dependent
attention, a bounded score residual, and soft acquisition progression. It consumes
frozen image embeddings and class scores and returns an explicit per-clip state.

The original Full search and common FSM settings are preserved in
`workflow/revision_20261013/targeted_hardneg_20261014/configs/search_full.json`
and `workflow/revision_20261013/targeted_hardneg_20261014/configs/fsm.json`.
These files accompany the search core; the existing incremental video entry
points remain separate. Experiment runners, ablation models, review artifacts,
and checkpoint files are not included in this source update.

## Verification

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Compatibility

Canonical entry points and implementation modules now use `anatomed` filenames.
Historical `*_anatopave.py` entry points and model/runtime modules are retained
as thin compatibility aliases. New checkpoints use `anatomed-joint-v1`; loaders
also accept the historical `anatopave-joint-v1` and `anatopave-joint-v2` formats.
Network parameter names and state-dict structure are unchanged.

CP-SET annotation IDs, class labels, masks and source-frame indices are independent
of the method name and are not renamed. Historical experiment artifacts and
checkpoint files are not rewritten by this source migration.

The observation loader accepts both legacy inline `main_text` labels and the
public path-only `text_path` format, reading the same TXT label content.
