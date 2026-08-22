#!/usr/bin/env bash
set -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

out="${ANATOPAVE_RUN_DIR:-runs/anatopave}"
log="$out/full_eval.log"

echo "[eval-start] best $(date -Is)" | tee "$log"
python eval_anatopave.py \
  --checkpoint "$out/best.pt" \
  --out_json "$out/eval_best_full.json" \
  --gen_max_new_tokens 320 \
  --bertscore_device cuda 2>&1 | tee -a "$log"
best_status=${PIPESTATUS[0]}
echo "[eval-exit] best code=$best_status $(date -Is)" | tee -a "$log"
if [[ $best_status -ne 0 ]]; then
  exit "$best_status"
fi

echo "[eval-start] last $(date -Is)" | tee -a "$log"
python eval_anatopave.py \
  --checkpoint "$out/resume_latest.pt" \
  --out_json "$out/eval_last_full.json" \
  --gen_max_new_tokens 320 \
  --bertscore_device cuda 2>&1 | tee -a "$log"
last_status=${PIPESTATUS[0]}
echo "[eval-exit] last code=$last_status $(date -Is)" | tee -a "$log"
exit "$last_status"
