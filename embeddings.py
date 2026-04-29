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

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from full NQ dataset.")
    parser.add_argument(
        "--dataset-name",
        default="sentence-transformers/natural-questions",
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
        default=512,
        help="Batch size for embedding answers",
    )
    parser.add_argument(
        "--collection-name",
        default="nq_full",
        help="ChromaDB collection name",
    )
    return parser.parse_args()


def build_index(
    dataset_name: str,
    embed_model_name: str,
    output_dir: Path,
    batch_size: int,
    collection_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name)
    dataset = raw["train"]
    print(f"Using full dataset rows: {len(dataset):,}")

    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)

    records: list[dict[str, str]] = []
    for i in range(len(dataset)):
        query = str(dataset[i]["query"]).strip()
        answer = str(dataset[i]["answer"]).strip()
        if not answer:
            continue
        records.append({"query": query, "answer": answer})

    if not records:
        raise RuntimeError("No valid records found in dataset.")

    chroma_dir = output_dir / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    docs = [r["answer"] for r in records]
    print(f"Encoding {len(docs):,} answers...")
    cfg_path = output_dir / "rag_config.json"
    n = len(records)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = records[start:end]
        batch_docs = [r["answer"] for r in batch]
        batch_embeddings = embedder.encode(
            batch_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        ids = [f"doc-{i}" for i in range(start, end)]
        metadatas = [{"query": r["query"]} for r in batch]
        collection.add(
            ids=ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        print(f"Indexed {end:,}/{n:,}")
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

