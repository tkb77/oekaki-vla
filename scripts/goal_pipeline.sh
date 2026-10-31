#!/usr/bin/env bash
# Training of the goal-image models on a separate GPU machine (see docs/cloud_training.md).
#
#   bash scripts/goal_pipeline.sh setup        # uv environment (Python 3.12, lerobot 0.6.1, torch + CUDA)
#   bash scripts/goal_pipeline.sh quickdraw    # download + preprocess Quick, Draw! (public, ~210 MB)
#   bash scripts/goal_pipeline.sh classifier   # judge CNN            (~10-20 min on one GPU)
#   bash scripts/goal_pipeline.sh generator    # class-conditional DDPM (100k steps, ~2-4 h)
#   bash scripts/goal_pipeline.sh overfit      # SmolVLA G on 30 episodes (pre-check, 4k steps, ~1 h); needs data/goal_v1
#   bash scripts/goal_pipeline.sh G            # SmolVLA model G (15k steps)
#   bash scripts/goal_pipeline.sh N            # SmolVLA model N (15k steps)
#   bash scripts/goal_pipeline.sh datagen_goal_v2   # dataset goal_v2: 2000 single + 500 two-drawing episodes (follow-up)
#   bash scripts/goal_pipeline.sh G_long|G_2x|G_pair   # follow-up G variants (configs/train_goal_G_*.yaml)
#   bash scripts/goal_pipeline.sh pack         # tar the trained weights for the trip back (outputs_goal.tar)
#
# Every stage logs to logs/goal_<stage>.log. Seeds are fixed in the configs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=${OEKAKI_PY:-"env -u PYTHONPATH .venv/bin/python"}   # scripts/cloud_task.sh points this at the image venv
stage=${1:?stage}
run() { echo "[$(date)] $stage: $*"; "$@" 2>&1 | tee "logs/goal_${stage}.log"; }
case "$stage" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync
    $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" ;;
  quickdraw)  run $PY scripts/download_quickdraw.py ;;
  classifier) run $PY scripts/train_classifier.py ;;
  generator)  run $PY scripts/train_generator.py ;;
  overfit)    run $PY scripts/train_smolvla.py --config configs/train_goal_overfit_G.yaml --dedicated ;;
  G)          run $PY scripts/train_smolvla.py --config configs/train_goal_G.yaml --dedicated ;;
  N)          run $PY scripts/train_smolvla.py --config configs/train_goal_N.yaml --dedicated ;;
  datagen_goal_v2) run $PY scripts/generate_goal_dataset.py --config configs/datagen_goal_v2.yaml ;;   # needs MuJoCo EGL (image >= 0.1.1)
  G_long|G_2x|G_pair) run $PY scripts/train_smolvla.py --config "configs/train_goal_${stage}.yaml" --dedicated ;;
  pack)
    files=()
    for f in outputs/classifier/judge.pt outputs/classifier/history.json outputs/generator/last.pt \
             outputs/generator/train_log.csv outputs/generator/samples_*.png; do [ -e "$f" ] && files+=("$f"); done
    for d in outputs/train/goal_overfit_G outputs/train/goal_G outputs/train/goal_N; do
      [ -d "$d/checkpoints/last" ] && files+=("$d/checkpoints/last" "$d/train_log.csv" "$d/run_info.json")
    done
    tar cf outputs_goal.tar "${files[@]}" logs/goal_*.log
    ls -la outputs_goal.tar ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
