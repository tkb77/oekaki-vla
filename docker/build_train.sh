#!/usr/bin/env bash
# Build -> smoke test -> push the training image (docker/Dockerfile.train).
#   REGISTRY=<registry/prefix> bash docker/build_train.sh <tag> [--no-push]
# A tag that already exists in the registry is never overwritten. Prints the pushed digest.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?tag, e.g. 0.1.0}
IMG=drawing_vla/smolvla_train:$TAG
docker build -f docker/Dockerfile.train -t "$IMG" --build-arg IMAGE_TAG="$IMG" \
    --build-arg BUILD_DATE="$(date -u +%FT%TZ)" --build-arg REPO_HEAD="$(git rev-parse --short HEAD)" .
docker run --rm --gpus all "$IMG" bash -lc 'nvidia-smi -L && aws --version && ffmpeg -version | head -1 &&
  python3 -c "import wandb; print(\"wandb\", wandb.__version__)" &&
  /opt/dvla/.venv/bin/python -c "import torch, lerobot; from torchcodec.decoders import VideoDecoder; print(\"torch\", torch.__version__, torch.cuda.is_available(), \"lerobot\", lerobot.__version__)"'
[ "${2:-}" = "--no-push" ] && exit 0
: "${REGISTRY:?REGISTRY (e.g. <registry host>/<namespace>)}"
if docker manifest inspect "$REGISTRY/$IMG" > /dev/null 2>&1; then echo "$REGISTRY/$IMG already exists; use a new tag"; exit 1; fi
docker tag "$IMG" "$REGISTRY/$IMG" && docker push "$REGISTRY/$IMG"
docker inspect --format '{{index .RepoDigests 0}}' "$REGISTRY/$IMG"
