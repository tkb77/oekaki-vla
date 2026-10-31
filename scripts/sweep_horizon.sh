#!/usr/bin/env bash
# Smoothness sweep on a 40-episode subset (4 per shape x standard/disturb): execution horizon (actions of each
# 50-step chunk executed before replanning) x sampling temperature (std of the initial flow-matching noise).
# Offline, the predicted chunks themselves are far jerkier than the expert's (sampling noise left in the
# chunk), so the temperature is swept together with the horizon.
# Usage: bash scripts/sweep_horizon.sh base|ttrtc|hermite|both     (waits for the model's last checkpoint)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
v=$1
ck=outputs/train/smooth_${v}/checkpoints/last
out=artifacts/eval_smooth_sweep
mkdir -p $out logs
until [ -f $ck/model.safetensors ]; do sleep 120; done

run() {  # tag, args...
  local tag=$1; shift
  [ -f "$out/results_${v}_${tag}.json" ] && return 0
  $PY scripts/evaluate.py --agent "${v}=$ck" --episodes-per-shape 4 --conditions standard disturb \
    --video-per-cell 0 --required-gpu-mib 11000 --out "$out/$tag" --tag "$tag" "$@" \
    > "logs/sweep_${v}_${tag}.log" 2>&1 && mv "$out/$tag/results_${v}_${tag}.json" "$out/"
}

if [[ $v == ttrtc || $v == both ]]; then
  for n in 1 0.5 0; do
    ( for s in 5 10 20; do run "ttrtc_d3_r${s}_n$n" --mode ttrtc --delay 3 --replan $s --noise-scale $n; done ) &
    sleep 120
  done
else
  ( run ensemble_n1 --mode ensemble
    for s in 5 10 20; do run "chunk_d0_r${s}_n1" --mode chunk --delay 0 --replan $s --noise-scale 1; done
    for s in 5 10; do run "rtc_d3_r${s}_n1" --mode rtc --delay 3 --replan $s --noise-scale 1; done ) &
  sleep 120
  ( for s in 5 10 20; do run "chunk_d0_r${s}_n0.5" --mode chunk --delay 0 --replan $s --noise-scale 0.5; done
    for s in 5 10; do run "rtc_d3_r${s}_n0.5" --mode rtc --delay 3 --replan $s --noise-scale 0.5; done ) &
  sleep 120
  ( for s in 5 10 20; do run "chunk_d0_r${s}_n0" --mode chunk --delay 0 --replan $s --noise-scale 0; done
    run ensemble_n0.5 --mode ensemble --noise-scale 0.5 ) &
fi
wait
echo "sweep $v done $(date)"
