"""Sharded retrieval: cosine vs shard rows uses OpenCL by default; optional NumPy fallback."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from paraverrag.config import (
    CPU_MAX_THREADS,
    ENCODER_MODEL,
    RETRIEVAL_ALLOW_NUMPY_COSINE,
    RETRIEVAL_CPU_WORKERS,
    RETRIEVAL_SHARD_DIR,
)

_MANIFEST = "shard_manifest.json"


def load_shard_manifest(shard_dir: str | Path | None = None) -> dict[str, Any]:
    p = Path(shard_dir or RETRIEVAL_SHARD_DIR) / _MANIFEST
    if not p.is_file():
        raise FileNotFoundError(
            f"Sharded retrieval manifest missing: {p}. "
            "Run ingest with POPULATE_WRITE_RETRIEVAL_SHARDS enabled (default) or set PARAVERAG_RETRIEVAL_SHARD_DIR."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _shard_paths(shard_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    files = manifest.get("shard_files") or []
    return [shard_dir / name for name in files]


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float32, order="C")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32, copy=False)


def _numpy_dot_scores(q: np.ndarray, emb_mmap: np.ndarray) -> np.ndarray:
    """Dot products (cosine if rows normalized). emb_mmap (n, dim)."""
    return np.asarray(emb_mmap @ q, dtype=np.float32)


def _topk_from_scores(scores: np.ndarray, docs: np.ndarray, k: int) -> list[tuple[float, str]]:
    if scores.size == 0 or k <= 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(float(scores[i]), str(docs[i])) for i in idx]


def _process_one_shard_numpy(args: tuple[Path, np.ndarray, int]) -> list[tuple[float, str]]:
    shard_path, q_norm, k_each = args
    z = np.load(shard_path, allow_pickle=True)
    emb = z["embeddings"]
    docs = z["documents"]
    scores = _numpy_dot_scores(q_norm, emb)
    return _topk_from_scores(scores, docs, k_each)


def _process_one_shard_opencl(args: tuple[Path, np.ndarray, int]) -> list[tuple[float, str]]:
    from paraverrag.retrieval.opencl_cosine import dot_scores_opencl

    shard_path, q_norm, k_each = args
    z = np.load(shard_path, allow_pickle=True)
    emb = np.asarray(z["embeddings"], dtype=np.float32, order="C")
    docs = z["documents"]
    scores = dot_scores_opencl(q_norm, emb)
    return _topk_from_scores(scores, docs, k_each)


def retrieve_from_shards(
    query_embedding: np.ndarray,
    k: int,
    *,
    shard_dir: str | Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Top-k document strings. Cosine similarity on shard rows uses **OpenCL** unless
    ``RETRIEVAL_ALLOW_NUMPY_COSINE`` is enabled (parallel NumPy path, capped by ``CPU_MAX_THREADS``).
    """
    sd = Path(shard_dir or RETRIEVAL_SHARD_DIR)
    man = manifest or load_shard_manifest(sd)
    paths = _shard_paths(sd, man)
    if not paths:
        return []

    enc_m = str(man.get("encoder_model") or "")
    if enc_m and enc_m != ENCODER_MODEL:
        print(
            f"[paraverrag] warning: shard manifest encoder {enc_m!r} != current {ENCODER_MODEL!r}",
            file=sys.stderr,
        )

    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if int(man.get("dim", q.shape[0])) != q.shape[0]:
        raise ValueError(f"Query dim {q.shape[0]} != manifest dim {man.get('dim')}")
    qn = _normalize_rows(q.reshape(1, -1))[0]

    k_each = max(k, 8)

    if not RETRIEVAL_ALLOW_NUMPY_COSINE:
        from paraverrag.retrieval.opencl_cosine import opencl_probe_error

        ocl_err = opencl_probe_error()
        if ocl_err is not None:
            raise RuntimeError(
                "Sharded retrieval is configured for OpenCL cosine similarity, but no working OpenCL "
                "platform was found.\n\n"
                f"Probe: {ocl_err}\n\n"
                "`pip install pyopencl` is not enough: you need an **OpenCL ICD** (vendor driver) so "
                "`clinfo` reports at least one platform. If `clinfo` shows **Number of platforms 0**, install e.g.:\n"
                "  • NVIDIA GPU (Ubuntu/Debian):  sudo apt install nvidia-opencl-icd\n"
                "  • Portable CPU OpenCL:         sudo apt install pocl-opencl-icd\n"
                "  • WSL2: GPU OpenCL depends on Windows + WSL GPU drivers; otherwise use POCL for CPU CL.\n\n"
                "Sharded cosine defaults to **CPU OpenCL** (PARAVERAG_OPENCL_DEVICE=cpu). With only a GPU ICD, "
                "set PARAVERAG_OPENCL_DEVICE=auto or gpu.\n\n"
                "Until a platform exists, use parallel CPU cosine: "
                "PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1\n"
                "Or use Chroma retrieval: PARAVERAG_RETRIEVAL_BACKEND=chroma"
            )
        merged: list[tuple[float, str]] = []
        for p in paths:
            merged.extend(_process_one_shard_opencl((p, qn, k_each)))
        merged.sort(key=lambda x: -x[0])
        return [t for _, t in merged[:k]]

    workers = max(1, min(RETRIEVAL_CPU_WORKERS, len(paths), CPU_MAX_THREADS))
    merged = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one_shard_numpy, (p, qn, k_each)): p for p in paths}
        for fut in as_completed(futs):
            merged.extend(fut.result())
    merged.sort(key=lambda x: -x[0])
    return [t for _, t in merged[:k]]


def write_shard_manifest(
    shard_dir: Path,
    *,
    encoder_model: str,
    dim: int,
    shard_files: list[str],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "encoder_model": encoder_model,
        "dim": int(dim),
        "n_shards": len(shard_files),
        "shard_files": shard_files,
        "space": "cosine",
    }
    (shard_dir / _MANIFEST).write_text(json.dumps(meta, indent=2), encoding="utf-8")
