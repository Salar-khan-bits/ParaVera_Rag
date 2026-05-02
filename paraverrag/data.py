"""Load ayaani12/Project_Dataset from per-shard Parquet (mixed schemas on Hub break single load_dataset call)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import chromadb

from datasets import concatenate_datasets, load_dataset

from paraverrag.config import CHROMA_DB_PATH, COLLECTION_NAME

HF_PARQUET = "hf://datasets/ayaani12/Project_Dataset"


def load_eval_items_from_json(
    path: str | Path,
    *,
    max_items: int | None,
    seed: int,
) -> list[dict]:
    """Load evaluation items from a JSON file.

    Expects a JSON array of objects with ``question`` and ``answer`` (or ``ground_truth``).
    If ``max_items`` is set and more rows are valid, draws ``max_items`` without replacement using ``seed``.
    If ``max_items`` is None, every valid row is returned.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Eval JSON not found: {p.resolve()}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{p}: JSON root must be an array, got {type(raw).__name__}.")
    out: list[dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"{p}: item {i} must be an object, got {type(row).__name__}.")
        q = row.get("question")
        gold = row.get("answer")
        if gold is None:
            gold = row.get("ground_truth")
        if q is None or gold is None:
            raise ValueError(
                f"{p}: item {i} needs non-null 'question' and 'answer' (or 'ground_truth')."
            )
        qs, gs = str(q).strip(), str(gold).strip()
        if not qs or not gs:
            continue
        rec: dict[str, str] = {"question": qs, "ground_truth": gs}
        raw_id = row.get("query_id")
        if raw_id is None:
            raw_id = row.get("id")
        if raw_id is not None and str(raw_id).strip():
            rec["query_id"] = str(raw_id).strip()
        out.append(rec)
    if not out:
        raise ValueError(f"{p}: no non-empty question/answer pairs found.")
    if max_items is not None and len(out) > max_items:
        rng = random.Random(seed)
        out = rng.sample(out, max_items)
    return out


def _natural_questions():
    return load_dataset(
        "parquet",
        data_files=f"{HF_PARQUET}/natural-questions.parquet",
        split="train",
    )


def _trivia_shard(filename: str):
    raw = load_dataset("parquet", data_files=f"{HF_PARQUET}/{filename}", split="train")

    def norm(ex):
        a = ex["answer"]
        if isinstance(a, dict):
            text = a.get("value") or a.get("normalized_value") or ""
        else:
            text = str(a)
        return {"query": (ex["question"] or "").strip(), "answer": text.strip()}

    return raw.map(norm, remove_columns=raw.column_names)


def _wikimedia_shard(filename: str):
    raw = load_dataset("parquet", data_files=f"{HF_PARQUET}/{filename}", split="train")

    def norm(ex):
        return {"query": (ex["title"] or "").strip(), "answer": (ex["text"] or "").strip()}

    return raw.map(norm, remove_columns=raw.column_names)


def load_train_qa_dataset():
    """Full train corpus as a single Dataset with columns `query`, `answer`."""
    parts = [
        _natural_questions(),
        _trivia_shard("trivia_qa_1.parquet"),
        _trivia_shard("trivia_qa_2.parquet"),
        _wikimedia_shard("wikimedia_1.parquet"),
        _wikimedia_shard("wikimedia_2.parquet"),
    ]
    ds = concatenate_datasets(parts)
    ds = ds.filter(lambda r: len(r["query"]) > 0 and len(r["answer"]) > 0)
    return ds


def _norm_trivia_row(ex: dict[str, Any]) -> dict[str, str]:
    a = ex["answer"]
    if isinstance(a, dict):
        text = a.get("value") or a.get("normalized_value") or ""
    else:
        text = str(a)
    return {"query": (ex["question"] or "").strip(), "answer": text.strip()}


def _norm_wikimedia_row(ex: dict[str, Any]) -> dict[str, str]:
    return {"query": (ex["title"] or "").strip(), "answer": (ex["text"] or "").strip()}


def _iter_natural_streaming() -> Iterator[dict[str, str]]:
    ds = load_dataset(
        "parquet",
        data_files=f"{HF_PARQUET}/natural-questions.parquet",
        split="train",
        streaming=True,
    )
    for ex in ds:
        q = (ex.get("query") or ex.get("question") or "").strip()
        a = ex.get("answer")
        if isinstance(a, dict):
            text = a.get("value") or a.get("normalized_value") or ""
        else:
            text = str(a or "")
        yield {"query": q, "answer": text.strip()}


def _iter_trivia_streaming(filename: str) -> Iterator[dict[str, str]]:
    raw = load_dataset("parquet", data_files=f"{HF_PARQUET}/{filename}", split="train", streaming=True)
    for ex in raw:
        yield _norm_trivia_row(ex)


def _iter_wikimedia_streaming(filename: str) -> Iterator[dict[str, str]]:
    raw = load_dataset("parquet", data_files=f"{HF_PARQUET}/{filename}", split="train", streaming=True)
    for ex in raw:
        yield _norm_wikimedia_row(ex)


def _iter_train_qa_streaming(shard_order_seed: int) -> Iterator[dict[str, str]]:
    """Stream normalized rows from all shards; shard visit order is randomized (memory-safe)."""
    factories: list[Callable[[], Iterator[dict[str, str]]]] = [
        _iter_natural_streaming,
        lambda: _iter_trivia_streaming("trivia_qa_1.parquet"),
        lambda: _iter_trivia_streaming("trivia_qa_2.parquet"),
        lambda: _iter_wikimedia_streaming("wikimedia_1.parquet"),
        lambda: _iter_wikimedia_streaming("wikimedia_2.parquet"),
    ]
    order = list(range(len(factories)))
    random.Random(shard_order_seed).shuffle(order)
    for idx in order:
        yield from factories[idx]()


def build_eval_subset(n: int, seed: int, max_valid_rows: int | None = None) -> list[dict]:
    """Sample n QA pairs via reservoir sampling without materializing the full corpus."""
    rng = random.Random(seed)
    shard_order_seed = seed ^ 0x9E3779B9
    buf: list[dict] = []
    valid_idx = -1
    for row in _iter_train_qa_streaming(shard_order_seed):
        if not row["query"] or not row["answer"]:
            continue
        valid_idx += 1
        rec = {
            "question": row["query"],
            "ground_truth": row["answer"],
            "query_id": f"hub_stream_{valid_idx}",
        }
        if len(buf) < n:
            buf.append(rec)
        else:
            j = rng.randint(0, valid_idx)
            if j < n:
                buf[j] = rec
        if max_valid_rows is not None and valid_idx + 1 >= max_valid_rows:
            break
    return buf


def build_eval_subset_from_chroma(n: int, seed: int) -> list[dict]:
    """Sample n (question, answer) pairs from the populated Chroma collection (no Hub access).

    Requires ``documents`` = gold answers and metadata key ``question`` (see
    ``POPULATE_STORE_QUESTION_METADATA`` in config). Ids must be ``0``..``count-1`` as in ``paraverrag.populate``.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    if total == 0:
        return []
    n = min(n, total)
    rng = random.Random(seed)
    chosen = rng.sample(range(total), n)
    ids_str = [str(i) for i in chosen]
    res = col.get(ids=ids_str, include=["documents", "metadatas"])
    id_to_doc: dict[str, str] = {}
    id_to_meta: dict[str, dict[str, Any]] = {}
    for i, cid in enumerate(res["ids"]):
        doc = (res["documents"] or [None] * len(res["ids"]))[i]
        meta = (res["metadatas"] or [None] * len(res["ids"]))[i]
        id_to_doc[cid] = (doc or "").strip()
        id_to_meta[cid] = meta if isinstance(meta, dict) else {}

    missing_q: list[str] = []
    out: list[dict] = []
    for cid in ids_str:
        meta = id_to_meta.get(cid, {})
        q = (meta.get("question") or "").strip()
        gold = id_to_doc.get(cid, "").strip()
        if not q:
            missing_q.append(cid)
            continue
        if not gold:
            continue
        out.append({"question": q, "ground_truth": gold, "query_id": f"chroma_{cid}"})

    if missing_q:
        raise RuntimeError(
            "Chroma is missing `question` metadata for sampled ids (e.g. "
            f"{missing_q[:5]}{'...' if len(missing_q) > 5 else ''}). "
            "Re-run paraverrag.populate with POPULATE_STORE_QUESTION_METADATA=True in paraverrag.config."
        )
    if len(out) < n:
        raise RuntimeError(
            f"Only {len(out)} usable rows with non-empty question+answer from Chroma (asked {n})."
        )
    return out
