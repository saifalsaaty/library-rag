"""Shared helpers for the numbered scripts (run them from the project root)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from librag.config import load_config, path, results_path, set_seed  # noqa: E402
from librag.schema import Record, read_jsonl  # noqa: E402


def init(ap: argparse.ArgumentParser) -> tuple[dict, argparse.Namespace, logging.Logger]:
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    log_dir = path(cfg, "results_dir") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = Path(sys.argv[0]).stem
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")],
    )
    return cfg, args, logging.getLogger(name)


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    return ap


def load_records(cfg: dict) -> dict[str, Record]:
    p = path(cfg, "processed_dir") / "records.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} not found — run scripts/02_prepare_dataset.py first")
    return {r.id: r for r in read_jsonl(p)}


__all__ = ["ROOT", "parser", "init", "load_records", "path", "results_path"]
