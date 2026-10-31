#!/usr/bin/env bash
# Smooth-drawing showcase: plain SmolVLA (B, sampled chunks every 5 steps) vs the same model with smooth inference vs the
# final model (A + training-time RTC + elapsed time), on identical episodes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH MUJOCO_GL=egl nice -n 15 .venv/bin/python"
B=outputs/train/smooth_B/checkpoints/last; T=outputs/train/smooth_tttime/checkpoints/last
A=(--agent "naive=$B|mode=chunk,replan=5,delay=0,noise_scale=1|1. plain SmolVLA (joints only)"
   --agent "smooth=$B|mode=rtc,replan=30,delay=3,noise_scale=0|2. + smooth inference (RTC)"
   --agent "final=$T|mode=ttrtc,replan=30,delay=3,noise_scale=0|3. final: + contact, TT-RTC, time")
$PY scripts/render_video.py "${A[@]}" --cond standard --shapes star --ep 1 --out artifacts/videos_smooth
$PY scripts/render_video.py "${A[@]}" --cond standard --shapes square --ep 7 --out artifacts/videos_smooth
$PY scripts/render_video.py "${A[@]}" --cond disturb --shapes circle --ep 6 --out artifacts/videos_smooth
