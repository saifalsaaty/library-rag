"""Configuration loading and project paths."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(cfg_path.resolve().parent)
    return cfg


def path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the project root and create its parent."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    target_dir = p if p.suffix == "" else p.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return p


def results_path(cfg: dict[str, Any], *parts: str) -> Path:
    p = path(cfg, "results_dir").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def read_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
