#!/usr/bin/env bash
# Smooth-drawing main evaluation (10 episodes per shape x standard/disturb = 100 episodes, one video per cell).
# All smooth schemes use the setting picked by the sweep (scripts/sweep_horizon.sh, sweep_more.sh): initial noise 0
# and a new chunk every 30 of its 50 steps. "naive" (noise 1, replan 5) and "ensemble" (noise 1, ACT temporal
# ensembling) are the settings used before the smoothing work. Each stream waits for its model's last checkpoint; every
# evaluate.py call checks nvidia-smi itself (--required-gpu-mib) and runs at nice 15.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts/eval_smooth
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
S="--replan 30 --noise-scale 0"

evaluate() {  # $1 = model variant, $2 = tag, rest = inference args
  local v=$1 tag=$2; shift 2
  local out=artifacts/eval_smooth ck=outputs/train/smooth_${v}/checkpoints/last
  [ -f "$out/results_${v}_${tag}.json" ] && return 0
  until [ -f $ck/model.safetensors ]; do sleep 120; done
  $PY scripts/evaluate.py --agent "${v}=$ck" --episodes-per-shape 10 \
    --conditions standard disturb --video-per-cell 1 --required-gpu-mib 11000 --out "$out/$tag" --tag "$tag" "$@" \
    > "logs/eval_smooth_${v}_${tag}.log" 2>&1 && mv "$out/$tag/results_${v}_${tag}.json" "$out/results_${v}_${tag}.json"
}

(
  evaluate base rtc_d3 --mode rtc --delay 3 $S
  evaluate base chunk_d0 --mode chunk --delay 0 $S
  evaluate base chunk_d3 --mode chunk --delay 3 $S
  evaluate base naive --mode chunk --delay 0 --replan 5 --noise-scale 1
  evaluate base ensemble --mode ensemble
) &
sleep 120
(
  evaluate ttrtc ttrtc_d3 --mode ttrtc --delay 3 $S
  evaluate hermite rtc_d3 --mode rtc --delay 3 $S
  evaluate hermite chunk_d0 --mode chunk --delay 0 $S
) &
sleep 120
(
  evaluate time rtc_d3 --mode rtc --delay 3 $S
  evaluate time chunk_d0 --mode chunk --delay 0 $S
  evaluate both ttrtc_d3 --mode ttrtc --delay 3 $S
) &
wait
echo "smooth main eval done $(date)"
