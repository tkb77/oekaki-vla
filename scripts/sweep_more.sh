#!/usr/bin/env bash
# Extra points for the smoothness sweep (same 40-episode subset as sweep_horizon.sh), run sequentially.
# Usage: bash scripts/sweep_more.sh <model> "<tag> <evaluate.py args...>" ...
set -uo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
v=$1; shift
ck=outputs/train/smooth_${v}/checkpoints/last
out=artifacts/eval_smooth_sweep
mkdir -p $out logs
until [ -f $ck/model.safetensors ]; do sleep 120; done
for spec in "$@"; do
  read -r tag args <<< "$spec"
  [ -f "$out/results_${v}_${tag}.json" ] && continue
  $PY scripts/evaluate.py --agent "${v}=$ck" --episodes-per-shape 4 --conditions standard disturb \
    --video-per-cell 0 --required-gpu-mib 11000 --out "$out/$tag" --tag "$tag" $args \
    > "logs/sweep_${v}_${tag}.log" 2>&1 && mv "$out/$tag/results_${v}_${tag}.json" "$out/"
done
echo "sweep_more $v done $(date)"
