"""Reading/writing queries, TREC runs and qrels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_QUERY_COLUMNS = ["qid", "query", "lang", "type", "answerable"]


def read_queries(p: str | Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Create it from data/queries/queries_template.xlsx "
            "following docs/annotation_guidelines.md (queries must come from librarians or real search logs)."
        )
    df = pd.read_excel(p, dtype=str) if p.suffix.lower() in (".xlsx", ".xls") else \
        pd.read_csv(p, dtype=str, sep="\t" if p.suffix == ".tsv" else ",", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_QUERY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"queries file is missing columns: {missing}")
    df = df.dropna(subset=["qid", "query"])
    df["qid"] = df["qid"].astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()
    df["lang"] = df["lang"].astype(str).str.strip().str.lower()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["answerable"] = df["answerable"].astype(str).str.strip().isin(["1", "yes", "true", "y", "نعم"])
    if df["qid"].duplicated().any():
        raise ValueError("duplicate qid values in queries file")
    return df.set_index("qid", drop=False)


def write_run(run: dict, p: str | Path, tag: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for qid in sorted(run):
            for rank, (doc, score) in enumerate(run[qid], start=1):
                fh.write(f"{qid} Q0 {doc} {rank} {score:.6f} {tag}\n")


def read_run(p: str | Path) -> dict:
    run: dict[str, list[tuple[str, float]]] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            qid, _, doc, _, score, _ = line.split()
            run.setdefault(qid, []).append((doc, float(score)))
    return run


def write_qrels(qrels: dict, p: str | Path) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for qid in sorted(qrels):
            for doc in sorted(qrels[qid]):
                fh.write(f"{qid} 0 {doc} {int(qrels[qid][doc])}\n")


def read_qrels(p: str | Path) -> dict:
    qrels: dict[str, dict[str, int]] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                qid, _, doc, rel = line.split()
                qrels.setdefault(qid, {})[doc] = int(rel)
    return qrels
