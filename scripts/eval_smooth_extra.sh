#!/usr/bin/env bash
# Main evaluation (100 episodes, same setting as run_smooth_eval.sh) for variants added later.
# Usage: bash scripts/eval_smooth_extra.sh <model> <tag> <evaluate.py args...>
set -uo pipefail
cd "$(dirname "$0")/.."
v=$1 tag=$2; shift 2
ck=outputs/train/smooth_${v}/checkpoints/last out=artifacts/eval_smooth
[ -f "$out/results_${v}_${tag}.json" ] && exit 0
until [ -f $ck/model.safetensors ]; do sleep 120; done
env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python scripts/evaluate.py --agent "${v}=$ck" --episodes-per-shape 10 \
  --conditions standard disturb --video-per-cell 1 --required-gpu-mib 11000 --out "$out/$tag" --tag "$tag" "$@" \
  > "logs/eval_smooth_${v}_${tag}.log" 2>&1 && mv "$out/$tag/results_${v}_${tag}.json" "$out/results_${v}_${tag}.json"
