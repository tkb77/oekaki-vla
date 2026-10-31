#!/usr/bin/env bash
# Goal-image closed-loop evaluation E1-E5 on a rented cloud GPU node (image docker/Dockerfile.train >= 0.1.1).
# Runs one scripts/evaluate_goal.py process per (setting, category), PER_GPU of them on each GPU at a time.
# Episodes are seeded per (setting, category, i), so the shards together are exactly the single-process run;
# scripts/merge_eval_shards.py joins them into outputs/eval_goal/.
#
#   DV_DATA=s3://<bucket>/data/drawing_vla DV_OUT=s3://<bucket>/checkpoints/drawing_vla/runs/<task> \
#       DV_MODELS=s3://.../models_goal.tar bash scripts/cloud_eval_goal.sh [PER_GPU=6] [extra evaluate_goal.py args]
#
# models_goal.tar holds outputs/train/goal_{G,N}/checkpoints/last, outputs/generator/last.pt (+ config.json) and
# outputs/classifier/judge.pt. Shards go to outputs/eval_goal_shards/<setting>_<category>/, synced to $DV_OUT.
#
# Optional environment:
#   EVAL_SETTINGS="E1 E2 E5"            settings to run (default all five)
#   EVAL_AGENTS="G_pair=outputs/..."     agents for every setting instead of the default G / N / expert sets
#   EVAL_FETCH="s3://...=local/dir ..."  extra checkpoints to download first (aws s3 sync)
#   EVAL_WAIT=s3://.../done/<task>.rc    wait until this object exists (e.g. the training task has finished)
# Uses the GPUs in CUDA_VISIBLE_DEVICES if it is set (one worker slot), otherwise every GPU of the node.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DV_DATA:?}" "${DV_OUT:?}" "${DV_MODELS:?}"
PER_GPU=${1:-6}; shift || true
EXTRA=("$@")
say() { echo "[$(date '+%m-%d %H:%M:%S')] cloud_eval: $*"; }
if [ -x "${DV_VENV:-/nonexistent}/bin/python" ]; then PY="$DV_VENV/bin/python"; export PYTHONPATH=$PWD
else PY=.venv/bin/python; fi
export MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p outputs/eval_goal_shards logs/eval_shards

sync_out() { aws s3 sync outputs/eval_goal_shards/ "$DV_OUT/eval_goal_shards/" --only-show-errors || true
             aws s3 sync logs/ "$DV_OUT/logs/" --only-show-errors || true; }
( while sleep 600; do sync_out; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_out; say "synced to $DV_OUT"' EXIT

[ -d data/quickdraw/processed ] || aws s3 cp "$DV_DATA/quickdraw_processed.tar" - --only-show-errors | tar x
aws s3 cp "$DV_MODELS" - --only-show-errors | tar x
if [ -n "${EVAL_WAIT:-}" ]; then
    say "waiting for $EVAL_WAIT"
    until aws s3 ls "$EVAL_WAIT" > /dev/null 2>&1; do sleep 60; done
fi
for f in ${EVAL_FETCH:-}; do aws s3 sync "${f%%=*}" "${f#*=}" --only-show-errors; done
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); fi
NG=${#GPUS[@]}
say "GPUs ${GPUS[*]}, $PER_GPU processes per GPU"

G=G=outputs/train/goal_G/checkpoints/last; N=N=outputs/train/goal_N/checkpoints/last
agents_for() { if [ -n "${EVAL_AGENTS:-}" ]; then for a in $EVAL_AGENTS; do echo -n "--agent $a "; done; return; fi
               case $1 in E1) echo "--agent $G --agent $N --agent expert=expert" ;;
                          E2|E5) echo "--agent $G" ;;
                          *) echo "--agent $G --agent $N" ;; esac; }
jobs=()
for s in ${EVAL_SETTINGS:-E1 E3 E4 E2 E5}; do
    for c in circle square triangle star "smiley face"; do jobs+=("$s|$c"); done
done

run_one() {  # index "setting|category"
    local i=$1 s=${2%%|*} c=${2#*|} tag
    tag="${s}_${c// /_}"
    local g=${GPUS[$(( i % NG ))]}
    say "start $tag on GPU $g"
    CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g nice -n 5 $PY scripts/evaluate_goal.py \
        $(agents_for "$s") --settings "$s" --categories "$c" --out "outputs/eval_goal_shards/$tag" \
        --required-gpu-mib 0 "${EXTRA[@]}" > "logs/eval_shards/$tag.log" 2>&1 \
        && say "done $tag" || say "FAILED $tag (rc $?)"
}
pids=(); i=0
for j in "${jobs[@]}"; do
    while [ "$(jobs -rp | grep -vcx "$SYNC_PID")" -ge $(( NG * PER_GPU )) ]; do sleep 5; done
    run_one "$i" "$j" &
    pids+=($!); i=$(( i + 1 ))
done
for p in "${pids[@]}"; do wait "$p" || true; done
n_ok=$(grep -l "^saved" logs/eval_shards/*.log 2>/dev/null | wc -l)
say "finished: $n_ok / ${#jobs[@]} shards saved"
[ "$n_ok" -eq "${#jobs[@]}" ]
