#!/usr/bin/env bash
# Smooth drawing: smoother action chunks. Trains the four variants on the fixed-floor dataset (data/fixed_floor/A)
# and evaluates them under several inference schemes. Every python job checks nvidia-smi itself
# (start only when its footprint + a margin for other users is free; memory cap; OOM back-off) and
# runs at low CPU priority. At most two trainings run at once and they are started 10 min apart, so the
# second one sees the memory already taken by the first.
# Usage: bash scripts/run_smooth.sh [train|eval|all]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
what=${1:-all}

train() {  # $1 = variant
  [ -f "outputs/train/smooth_$1/checkpoints/last/model.safetensors" ] && return 0
  $PY scripts/train_smolvla.py --config "configs/train_smooth_$1.yaml" 2>&1 | grep --line-buffered -v "HTTP Request" \
    > "logs/train_smooth_$1.log"
}

evaluate() {  # $1 = model variant, $2 = tag, rest = inference args
  local v=$1 tag=$2; shift 2
  local out=artifacts/eval_smooth
  [ -f "$out/results_${v}_${tag}.json" ] && return 0
  $PY scripts/evaluate.py --agent "${v}=outputs/train/smooth_${v}/checkpoints/last" --episodes-per-shape 10 \
    --conditions standard disturb --video-per-cell 1 --required-gpu-mib 11000 --out "$out/$tag" --tag "$tag" "$@" \
    > "logs/eval_smooth_${v}_${tag}.log" 2>&1 && mv "$out/$tag/results_${v}_${tag}.json" "$out/results_${v}_${tag}.json"
}

if [[ $what == train || $what == all ]]; then
  until [ -f data/fixed_floor/A/meta/info.json ] && ! pgrep -f "generate_dataset.py --name fixed_floor" >/dev/null; do sleep 60; done
  (train base; train hermite) &
  sleep 600
  (train ttrtc; train both) &
  wait
fi

if [[ $what == eval || $what == all ]]; then
  mkdir -p artifacts/eval_smooth
  [ -f artifacts/eval_smooth/results_expert_expert.json ] || \
    $PY scripts/evaluate.py --agent expert=expert --episodes-per-shape 10 --conditions standard disturb \
      --video-per-cell 1 --no-clip --required-gpu-mib 0 --out artifacts/eval_smooth/expert --tag expert \
      > logs/eval_smooth_expert.log 2>&1 && mv artifacts/eval_smooth/expert/results_expert_expert.json artifacts/eval_smooth/
  # two evaluation streams in parallel (each ~2-3 GiB GPU)
  (
    evaluate base ensemble --mode ensemble
    evaluate base chunk_d0 --mode chunk --delay 0
    evaluate base chunk_d3 --mode chunk --delay 3
    evaluate base rtc_d3 --mode rtc --delay 3
    evaluate hermite ensemble --mode ensemble
  ) &
  sleep 300
  (
    evaluate ttrtc ttrtc_d3 --mode ttrtc --delay 3
    evaluate both ttrtc_d3 --mode ttrtc --delay 3
    evaluate hermite chunk_d0 --mode chunk --delay 0
    evaluate hermite chunk_d3 --mode chunk --delay 3
    evaluate hermite rtc_d3 --mode rtc --delay 3
  ) &
  wait
fi
echo "smooth $what done $(date)"
