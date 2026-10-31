#!/usr/bin/env bash
# Chains evaluation -> report -> videos once the training checkpoints appear.
# A is evaluated while B is still training (inference needs only ~2 GB).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
wait_ckpt() { while [ ! -f "outputs/train/$1/checkpoints/last/model.safetensors" ]; do sleep 120; done; sleep 60; }
evaluate() {
  local c=$1
  nice -n 5 $PY scripts/evaluate.py --agent "$c=outputs/train/$c/checkpoints/last" --episodes-per-shape 10 \
      --conditions standard disturb --out artifacts/eval --tag main
  nice -n 5 $PY scripts/evaluate.py --agent "$c=outputs/train/$c/checkpoints/last" --episodes-per-shape 4 \
      --conditions out_of_range --out artifacts/eval --tag oor --video-per-cell 0
}
wait_ckpt A; echo "A done $(date)"; evaluate A
wait_ckpt B; echo "B done $(date)"; evaluate B
$PY scripts/make_report.py --eval artifacts/eval/results_A_main.json artifacts/eval/results_B_main.json \
    artifacts/eval/results_A_oor.json artifacts/eval/results_B_oor.json artifacts/eval/results_expert.json \
    --out artifacts/report
for s in star circle square; do
  nice -n 5 $PY scripts/render_video.py --agent A=outputs/train/A/checkpoints/last \
      --agent B=outputs/train/B/checkpoints/last --cond disturb --shapes $s --ep 0 --out artifacts/videos
done
echo "pipeline finished $(date)"
