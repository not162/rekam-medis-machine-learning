"""Small shared helpers: config loading, seeding, atomic JSON IO."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge ``upd`` into ``base`` (in place) and return ``base``."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "training.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def write_json(path: str | Path, obj) -> None:
    """Atomic JSON write (tmp file + replace) so interrupts never leave a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
