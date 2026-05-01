"""
Build ChromaDB collection for Natural Questions.

Usage:
  python embeddings.py
  python embeddings.py --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import chromadb
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from full project dataset.")
    parser.add_argument(
        "--dataset-name",
        default="ayaani12/Project_Dataset",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where ChromaDB files and config are saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for embedding answers",
    )
    parser.add_argument(
        "--collection-name",
        default="nq_full",
        help="ChromaDB collection name",
    )
    return parser.parse_args()


def _extract_answer(row: dict[str, Any]) -> str:
    answer_raw = row.get("answer")
    if isinstance(answer_raw, dict):
        return str(
            answer_raw.get("value")
            or answer_raw.get("normalized_value")
            or (answer_raw.get("aliases") or [""])[0]
        ).strip()
    return str(
        answer_raw
        or row.get("response")
        or row.get("output")
        or row.get("text")
        or ""
    ).strip()


def _load_project_answers(dataset_name: str) -> list[str]:
    parquet_files = [
        path
        for path in list_repo_files(repo_id=dataset_name, repo_type="dataset")
        if path.endswith(".parquet")
    ]
    answers: list[str] = []
    for parquet_path in sorted(parquet_files):
        print(f"[embeddings] Reading {parquet_path}")
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=parquet_path,
        )
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for item in batch.to_pylist():
                answer = _extract_answer(dict(item))
                if answer:
                    answers.append(answer)
    return answers


def build_index(
    dataset_name: str,
    embed_model_name: str,
    output_dir: Path,
    batch_size: int,
    collection_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    answers = _load_project_answers(dataset_name)
    print(f"Loaded total answer rows: {len(answers):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {embed_model_name} on {device}")
    embedder = SentenceTransformer(embed_model_name, device=device)

    # Store only answer texts for indexing. Questions are not embedded.
    if not answers:
        raise RuntimeError("No valid records found in dataset.")

    chroma_dir = output_dir / "chroma_db"
    if chroma_dir.exists():
        print(f"[embeddings] Removing existing vector DB: {chroma_dir}")
        shutil.rmtree(chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    docs = answers
    print(f"Encoding {len(docs):,} answers...")
    cfg_path = output_dir / "rag_config.json"
    n = len(answers)
    active_batch_size = max(1, batch_size)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_docs = answers[start:end]
        while True:
            try:
                batch_embeddings = embedder.encode(
                    batch_docs,
                    batch_size=active_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).tolist()
                break
            except torch.cuda.OutOfMemoryError:
                if device == "cuda" and active_batch_size > 1:
                    torch.cuda.empty_cache()
                    new_batch = max(1, active_batch_size // 2)
                    if new_batch == active_batch_size:
                        new_batch = 1
                    print(
                        f"[embeddings] CUDA OOM at batch_size={active_batch_size}, retrying with batch_size={new_batch}",
                        flush=True,
                    )
                    active_batch_size = new_batch
                    continue
                if device == "cuda":
                    print(
                        "[embeddings] CUDA OOM persisted at batch_size=1. Falling back to CPU.",
                        flush=True,
                    )
                    device = "cpu"
                    embedder = SentenceTransformer(embed_model_name, device=device)
                    active_batch_size = max(16, min(batch_size, 256))
                    continue
                raise
        ids = [f"doc-{i}" for i in range(start, end)]
        metadatas = [{"source": "answer_only"} for _ in range(start, end)]
        collection.add(
            ids=ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        print(f"Indexed {end:,}/{n:,} (batch_size={active_batch_size}, device={device})")
    print(f"ChromaDB collection built: {collection_name}")

    cfg = {
        "dataset_name": dataset_name,
        "embed_model": embed_model_name,
        "chroma_path": str(chroma_dir),
        "collection_name": collection_name,
    }
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config: {cfg_path}")


def main() -> None:
    args = parse_args()
    build_index(
        dataset_name=args.dataset_name,
        embed_model_name=args.embed_model,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()

