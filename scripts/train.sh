#!/usr/bin/env bash
# Train conditions A and B with identical data / steps / seed, one after the other.
# The python script checks nvidia-smi itself (waits for free memory, sizes the micro-batch).
# Usage: bash scripts/train.sh [extra args for train_smolvla.py, e.g. --steps 20000]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for cond in A B; do
  echo "=== condition ${cond} ===  $(date)"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
  nice -n 5 .venv/bin/python scripts/train_smolvla.py --config "configs/train_${cond}.yaml" "$@" \
    2>&1 | grep --line-buffered -v "HTTP Request" | tee "logs/train_${cond}.log"
done
