"""Dense retrieval: sentence-transformers encoders + exact cosine search.

Exact (brute-force) inner-product search over L2-normalised vectors is used in
all experiments so that results are not affected by approximate-nearest-
neighbour error. At tens of thousands of records this takes milliseconds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DenseEncoder:
    def __init__(self, model_id: str, device: str = "auto", max_seq_length_cap: int | None = 512,
                 batch_size: int = 64):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.cap = max_seq_length_cap
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_id, device=self.device)
            if self.cap and (self._model.max_seq_length or 10**9) > self.cap:
                self._model.max_seq_length = self.cap
        return self._model

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def encode(self, texts: list[str], prefix: str = "", show_progress: bool = False) -> np.ndarray:
        """Encode with automatic batch halving on CUDA out-of-memory (8 GB cards such as the RTX 4060)."""
        batch = self.batch_size
        while True:
            try:
                emb = self.model.encode([prefix + t for t in texts], batch_size=batch,
                                        normalize_embeddings=True, convert_to_numpy=True,
                                        show_progress_bar=show_progress)
                return emb.astype(np.float32)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch <= 1:
                    raise
                batch //= 2
                self.batch_size = batch
                self._empty_cache()
                print(f"CUDA out of memory — retrying {self.model_id} with batch_size={batch}")

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def release(self) -> None:
        """Unload the model and free GPU memory."""
        self._model = None
        import gc

        gc.collect()
        self._empty_cache()

    def token_lengths(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Untruncated token counts, used to report how much text each model silently drops."""
        tok = self.model.tokenizer
        out = []
        for i in range(0, len(texts), 1024):
            enc = tok([prefix + t for t in texts[i:i + 1024]], add_special_tokens=True, truncation=False)
            out.extend(len(x) for x in enc["input_ids"])
        return np.asarray(out)


class ExactIndex:
    def __init__(self, doc_ids: list[str], embeddings: np.ndarray):
        self.doc_ids = list(doc_ids)
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)

    def search_vectors(self, qvecs: np.ndarray, k: int = 100) -> list[list[tuple[str, float]]]:
        qvecs = np.atleast_2d(qvecs).astype(np.float32)
        sims = qvecs @ self.emb.T
        k = min(k, sims.shape[1])
        out = []
        idx_all = np.arange(sims.shape[1])
        for row in sims:
            part = np.argpartition(-row, k - 1)[:k] if k < row.size else idx_all
            order = part[np.lexsort((part, -row[part]))][:k]
            out.append([(self.doc_ids[i], float(row[i])) for i in order])
        return out

    def save(self, d: str | Path, meta: dict | None = None) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.emb)
        (d / "doc_ids.json").write_text(json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8")
        if meta is not None:
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: str | Path) -> "ExactIndex":
        d = Path(d)
        ids = json.loads((d / "doc_ids.json").read_text(encoding="utf-8"))
        return cls(ids, np.load(d / "embeddings.npy"))


def build_dense_index(encoder: DenseEncoder, doc_ids: list[str], texts: list[str], prefix: str,
                      out_dir: str | Path) -> dict:
    """Encode all documents, save the index and return build statistics for the paper."""
    lengths = encoder.token_lengths(texts, prefix)
    t0 = time.perf_counter()
    emb = encoder.encode(texts, prefix, show_progress=True)
    secs = time.perf_counter() - t0
    meta = {
        "model_id": encoder.model_id,
        "device": encoder.device,
        "dims": int(emb.shape[1]),
        "n_docs": len(doc_ids),
        "max_seq_length": encoder.max_seq_length,
        "encode_seconds": round(secs, 2),
        "docs_per_second": round(len(doc_ids) / secs, 1) if secs else None,
        "truncated_pct": round(100.0 * float((lengths > encoder.max_seq_length).mean()), 2),
        "token_length_median": float(np.median(lengths)),
        "token_length_p95": float(np.percentile(lengths, 95)),
    }
    ExactIndex(doc_ids, emb).save(out_dir, meta)
    return meta
