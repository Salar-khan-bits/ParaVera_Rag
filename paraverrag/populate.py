#!/usr/bin/env python3
"""Populate Chroma from the HF corpus: CUDA-encode batches, parallel writer thread."""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import traceback
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from paraverrag.config import (
    CHROMA_DB_PATH,
    CHROMA_HNSW_CONSTRUCTION_EF,
    CHROMA_HNSW_INTERNAL_BATCH,
    CHROMA_HNSW_SYNC_THRESHOLD,
    COLLECTION_NAME,
    ENCODER_MODEL,
    EVAL_JSON_PATH,
    PERCENTAGE_TEST_DATA,
    POPULATE_ENCODE_BATCH_SIZE,
    POPULATE_MAX_ROWS,
    POPULATE_PIPELINE_QUEUE_DEPTH,
    POPULATE_STORE_QUESTION_METADATA,
    POPULATE_WRITE_EVAL_JSON,
    POPULATE_WRITE_RETRIEVAL_SHARDS,
    RANDOM_SEED,
    RETRIEVAL_SHARD_COUNT,
    RETRIEVAL_SHARD_DIR,
)
from paraverrag.retrieval.sharded_retrieve import write_shard_manifest
from paraverrag.data import load_train_qa_dataset


def _write_eval_json_from_ingested_slice(
    ds,
    n: int,
    *,
    out_path: str,
    pct: float,
    seed: int,
) -> None:
    """Sample ``pct``% of indices ``0..n-1`` (same rows as Chroma) into eval JSON."""
    pct_eff = min(100.0, max(0.0, float(pct)))
    if pct_eff <= 0:
        print("PERCENTAGE_TEST_DATA <= 0; skipping eval JSON.", flush=True)
        return
    # Use ceil so small corpora still get at least one eval row (int() would give 0 for e.g. n=9, 10%).
    test_n = min(n, math.ceil(n * pct_eff / 100.0)) if n > 0 else 0
    if test_n <= 0:
        print(
            f"PERCENTAGE_TEST_DATA={pct}% of n={n} yields 0 rows; skipping eval JSON.",
            flush=True,
        )
        return
    rng = random.Random(seed)
    idxs = rng.sample(range(n), test_n)
    rows: list[dict[str, str]] = []
    for i in idxs:
        row = ds[int(i)]
        q = str(row["query"]).strip()
        a = str(row["answer"]).strip()
        if q and a:
            rows.append({"question": q, "answer": a})
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(rows)} eval rows ({pct_eff:g}% of {n} ingested) to {path.resolve()}",
        flush=True,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "Warning: CUDA not available; using CPU for embeddings. "
            "The writer thread still overlaps I/O with encoding."
        )

    print("Loading ayaani12/Project_Dataset (all Parquet shards → query/answer)...")
    ds = load_train_qa_dataset()
    full_n = len(ds)
    if POPULATE_MAX_ROWS is not None:
        n = min(POPULATE_MAX_ROWS, full_n)
        if n <= 0:
            print("POPULATE_MAX_ROWS is 0 or negative; nothing to ingest.", flush=True)
            return
        if n < full_n:
            ds = ds.select(range(n))
            print(
                f"Ingesting first {n} of {full_n} rows (POPULATE_MAX_ROWS={POPULATE_MAX_ROWS}).",
                flush=True,
            )
        else:
            print(
                f"Ingesting all {n} rows (POPULATE_MAX_ROWS={POPULATE_MAX_ROWS} >= corpus size).",
                flush=True,
            )
    else:
        n = full_n
        print(f"Ingesting full corpus: {n} rows.", flush=True)

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ingest_batch = client.get_max_batch_size()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    sync_th = max(CHROMA_HNSW_SYNC_THRESHOLD, n + 10_000)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:sync_threshold": sync_th,
            "hnsw:batch_size": min(CHROMA_HNSW_INTERNAL_BATCH, sync_th - 1),
            "hnsw:construction_ef": CHROMA_HNSW_CONSTRUCTION_EF,
        },
    )

    enc_bs = min(POPULATE_ENCODE_BATCH_SIZE, ingest_batch)
    if device == "cpu":
        enc_bs = min(64, enc_bs)

    print(f"Loading encoder {ENCODER_MODEL} on {device}...")
    model = SentenceTransformer(ENCODER_MODEL, device=device)
    model.eval()

    qdepth = max(1, int(POPULATE_PIPELINE_QUEUE_DEPTH))
    num_batches = (n + ingest_batch - 1) // ingest_batch
    print(
        f"Pipelined ingest: n={n}, Chroma batch={ingest_batch}, encode micro-batch={enc_bs}, "
        f"queue_depth={qdepth}, hnsw:sync_threshold={sync_th}, "
        f"metadata={'on' if POPULATE_STORE_QUESTION_METADATA else 'off'}."
    )

    work_q: queue.Queue[tuple | None] = queue.Queue(maxsize=qdepth)
    writer_error: list[BaseException] = []

    def chroma_writer() -> None:
        try:
            while True:
                item = work_q.get()
                if item is None:
                    work_q.task_done()
                    break
                ids_b, emb_b, docs_b, metas_b = item
                kwargs: dict = {"ids": ids_b, "embeddings": emb_b, "documents": docs_b}
                if metas_b is not None:
                    kwargs["metadatas"] = metas_b
                collection.add(**kwargs)
                work_q.task_done()
        except BaseException as e:
            writer_error.append(e)
            traceback.print_exc()
            while True:
                try:
                    work_q.get_nowait()
                    work_q.task_done()
                except queue.Empty:
                    break

    writer = threading.Thread(target=chroma_writer, name="chroma-writer", daemon=True)
    writer.start()

    shard_dir = Path(RETRIEVAL_SHARD_DIR)
    n_shards = max(1, int(RETRIEVAL_SHARD_COUNT))
    shard_emb_lists: list[list[np.ndarray]] | None = (
        [[] for _ in range(n_shards)] if POPULATE_WRITE_RETRIEVAL_SHARDS else None
    )
    shard_doc_lists: list[list[str]] | None = (
        [[] for _ in range(n_shards)] if POPULATE_WRITE_RETRIEVAL_SHARDS else None
    )

    pbar = tqdm(total=num_batches, desc="encode (CUDA) + store")

    try:
        with torch.inference_mode():
            for start in range(0, n, ingest_batch):
                end = min(start + ingest_batch, n)
                shard = ds[start:end]
                batch_answers = [a.strip() for a in shard["answer"]]
                batch_questions = (
                    [q.strip() for q in shard["query"]]
                    if POPULATE_STORE_QUESTION_METADATA
                    else []
                )
                ids_b = [str(i) for i in range(start, end)]

                emb = model.encode(
                    batch_answers,
                    batch_size=enc_bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                emb = np.asarray(emb, dtype=np.float32, order="C")

                if shard_emb_lists is not None and shard_doc_lists is not None:
                    for j, global_idx in enumerate(range(start, end)):
                        sid = global_idx % n_shards
                        shard_emb_lists[sid].append(np.asarray(emb[j], dtype=np.float32, order="C"))
                        shard_doc_lists[sid].append(batch_answers[j])

                metas_b = None
                if POPULATE_STORE_QUESTION_METADATA:
                    metas_b = [{"question": q[:4000]} for q in batch_questions]

                work_q.put((ids_b, emb, batch_answers, metas_b))
                pbar.update(1)

        work_q.put(None)
        writer.join()
        pbar.close()

        if writer_error:
            raise writer_error[0]

    except BaseException:
        try:
            work_q.put(None, timeout=60)
        except queue.Full:
            print("Chroma writer queue full; writer may still be flushing.", flush=True)
        writer.join(timeout=300)
        pbar.close()
        raise

    if POPULATE_WRITE_EVAL_JSON:
        _write_eval_json_from_ingested_slice(
            ds,
            n,
            out_path=EVAL_JSON_PATH,
            pct=PERCENTAGE_TEST_DATA,
            seed=RANDOM_SEED,
        )

    if (
        n > 0
        and POPULATE_WRITE_RETRIEVAL_SHARDS
        and shard_emb_lists is not None
        and shard_doc_lists is not None
    ):
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_names: list[str] = []
        dim = 0
        for sid in range(n_shards):
            if shard_emb_lists[sid]:
                dim = int(np.asarray(shard_emb_lists[sid][0]).shape[0])
                break
        for sid in range(n_shards):
            if not shard_emb_lists[sid]:
                mat = np.zeros((0, dim), dtype=np.float32) if dim > 0 else np.zeros((0, 0), dtype=np.float32)
                doc_arr = np.array([], dtype=object)
            else:
                mat = np.stack(shard_emb_lists[sid], axis=0).astype(np.float32, order="C", copy=False)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                mat = (mat / norms).astype(np.float32, copy=False)
                doc_arr = np.asarray(shard_doc_lists[sid], dtype=object)
            name = f"shard_{sid:04d}.npz"
            np.savez_compressed(shard_dir / name, embeddings=mat, documents=doc_arr)
            shard_names.append(name)
        if dim <= 0 and n > 0:
            raise RuntimeError("Sharded export failed: no embedding dimension (empty corpus?).")
        if dim > 0:
            write_shard_manifest(
                shard_dir,
                encoder_model=ENCODER_MODEL,
                dim=dim,
                shard_files=shard_names,
            )
            print(
                f"Wrote {len(shard_names)} retrieval shards under {shard_dir.resolve()} "
                f"(set PARAVERAG_RETRIEVAL_BACKEND=sharded to use parallel CPU / OpenCL search).",
                flush=True,
            )

    del ds, model
    if device == "cuda":
        torch.cuda.empty_cache()

    count = collection.count()
    print(f"Done. Collection '{COLLECTION_NAME}' has {count} documents at {CHROMA_DB_PATH}.")


if __name__ == "__main__":
    main()
