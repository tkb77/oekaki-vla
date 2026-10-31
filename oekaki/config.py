"""Config loading helpers."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/env.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve_ckpt(spec: str | Path, filename: str | None = None) -> Path:
    """Local path, or a Hugging Face Hub repo id ("user/name"), downloaded (and cached) on first use.
    `filename` picks a file inside the downloaded repo (e.g. "last.pt" for the generator)."""
    path = resolve(spec)
    if path.exists() or re.fullmatch(r"[\w.-]+/[\w.-]+", str(spec)) is None:
        return path
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(str(spec)))
    return local / filename if filename else local
