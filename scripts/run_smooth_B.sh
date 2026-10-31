#!/usr/bin/env bash
# Smooth drawing, condition B (joints only) for the A/B x inference 2x2, then the remaining variants. Runs one training at a
# time (host RAM on this machine is shared and was exhausted once by the combined load).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
train() {
  [ -f "outputs/train/smooth_$1/checkpoints/last/model.safetensors" ] && return 0
  $PY scripts/train_smolvla.py --config "configs/train_smooth_$1.yaml" 2>&1 | grep --line-buffered -v "HTTP Request" \
    > "logs/train_smooth_$1.log"
}
S="--replan 30 --noise-scale 0"
train B
bash scripts/eval_smooth_extra.sh B naive --mode chunk --delay 0 --replan 5 --noise-scale 1
bash scripts/eval_smooth_extra.sh B rtc_d3 --mode rtc --delay 3 $S
bash scripts/eval_smooth_extra.sh B ensemble --mode ensemble
bash scripts/eval_smooth_extra.sh B chunk_d0 --mode chunk --delay 0 $S
train tttime
bash scripts/eval_smooth_extra.sh tttime ttrtc_d3 --mode ttrtc --delay 3 $S
bash scripts/eval_smooth_extra.sh tttime chunk_d0 --mode chunk --delay 0 $S
bash scripts/eval_smooth_extra.sh hermite1 rtc_d3 --mode rtc --delay 3 $S
echo "smooth B chain done $(date)"
