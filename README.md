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

## Dataset access

CP-SET is available for non-commercial research and education under the
CP-SET Non-Commercial Research Data License V1.0. To request access and accept
the applicable terms, please contact yangfeiyang@jlu.edu.cn.

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

## Full historical search core

`revision_batch1/search.py` provides the `AnatomicalSearch` module used by the
Full historical-search experiments: causal four-role memory, candidate-dependent
attention, a bounded score residual, and soft acquisition progression. It consumes
frozen image embeddings and class scores and returns an explicit per-clip state.

## Verification

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
