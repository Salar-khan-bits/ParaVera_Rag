"""ParaVerRAG main entrypoint.

Modes:
1) --initdb: load full dataset and build vector DB index only.
2) default: query existing vector DB (assumes index already exists).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

from embeddings import build_index
from rag_cli import load_config, retrieve

import chromadb
from sentence_transformers import SentenceTransformer
from config import DEFAULT_CONFIG
from correction.self_corrector import run_self_correction_loop
from generation.llm_generator import LlmGenerator
from scoring.gpu_scorer import GpuScorer


def _log(message: str) -> None:
    print(f"[main] {message}", flush=True)


_EMBEDDER_CACHE: Dict[str, SentenceTransformer] = {}
_COLLECTION_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_embedder(embed_model_name: str) -> SentenceTransformer:
    embedder = _EMBEDDER_CACHE.get(embed_model_name)
    if embedder is None:
        _log(f"Loading embedding model once: {embed_model_name}")
        embedder = SentenceTransformer(embed_model_name)
        _EMBEDDER_CACHE[embed_model_name] = embedder
    return embedder


def _get_collection(chroma_path: Path, collection_name: str) -> Any:
    key = (str(chroma_path), collection_name)
    collection = _COLLECTION_CACHE.get(key)
    if collection is None:
        _log(f"Connecting to ChromaDB once: {chroma_path}")
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(collection_name)
        _COLLECTION_CACHE[key] = collection
    return collection


def run_pipeline(
    query: str,
    cfg: Dict[str, Any],
    llm_url: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    top_k: int,
) -> dict[str, object]:
    start = perf_counter()
    chroma_path = Path(cfg["chroma_path"])
    collection_name = str(cfg["collection_name"])
    embed_model_name = str(cfg["embed_model"])

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found: {chroma_path}. Run `python main.py --initdb` first."
        )

    embedder = _get_embedder(embed_model_name)
    collection = _get_collection(chroma_path, collection_name)

    _log(f"Retrieving top-{top_k} documents")
    hits = retrieve(query, embedder, collection, top_k)
    retrieved_docs = [
        {"text": row["answer"], "retrieval_score": score, "question": row["query"]}
        for row, score in hits
    ]
    if not retrieved_docs:
        return {
            "query": query,
            "answer": "",
            "verified": False,
            "attempts": 0,
            "verification": {"all_passed": False, "results": [], "feedback": "No retrieval hits found."},
            "retrieved_docs": [],
            "latency_seconds": perf_counter() - start,
            "message": "No retrieval hits found.",
        }

    # Keep global config object but override runtime LLM args.
    runtime_cfg = DEFAULT_CONFIG.__class__(
        **{
            **DEFAULT_CONFIG.__dict__,
            "llm_api_url": llm_url,
            "generator_model_name": llm_model,
            "llm_temperature": temperature,
            "llm_max_tokens": max_tokens,
            "llm_timeout_seconds": float(timeout),
            "use_remote_llm_api": True,
        }
    )
    scored_docs = GpuScorer(runtime_cfg).score(retrieved_docs)
    generator = LlmGenerator(runtime_cfg)
    corrected = run_self_correction_loop(query, scored_docs, generator, runtime_cfg)

    return {
        "query": query,
        "answer": corrected["answer"],
        "verified": corrected["verified"],
        "attempts": corrected["attempts"],
        "verification": corrected["verification"],
        "retrieved_docs": scored_docs,
        "latency_seconds": perf_counter() - start,
    }


def run_query(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(Path(args.config))
    if args.chroma_path:
        cfg["chroma_path"] = args.chroma_path
    if args.collection_name:
        cfg["collection_name"] = args.collection_name
    return run_pipeline(
        query=args.query,
        cfg=cfg,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        top_k=args.top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ParaVerRAG main entrypoint.")
    parser.add_argument("query_words", nargs="*", help="Question to answer in query mode.")
    parser.add_argument(
        "--initdb",
        action="store_true",
        help="Initialize vector DB from full dataset and exit.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ayaani12/Project_Dataset",
        help="Hugging Face dataset name used for --initdb",
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model used for --initdb and query mode",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for index/config")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size used in --initdb mode")
    parser.add_argument("--collection-name", default="nq_full", help="Collection name override")
    parser.add_argument("--config", default="artifacts/rag_config.json", help="RAG config path")
    parser.add_argument("--chroma-path", default="", help="Override ChromaDB path")
    parser.add_argument("--llm-url", default="http://localhost:8080/v1/chat/completions", help="LLM endpoint")
    parser.add_argument("--llm-model", default="local-model", help="LLM model name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved docs")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=300, help="LLM max tokens")
    parser.add_argument("--timeout", type=int, default=120, help="LLM timeout in seconds")
    args = parser.parse_args()

    if args.initdb:
        _log("Running initdb mode: load + encode full dataset, then exit")
        build_index(
            dataset_name=args.dataset_name,
            embed_model_name=args.embed_model,
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            collection_name=args.collection_name,
        )
        _log("initdb complete. Pipeline query mode was not executed.")
        return

    query = " ".join(args.query_words).strip()
    if query:
        args.query = query
        _log("Running full pipeline query mode")
        result = run_query(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    _log("No query provided. Entering interactive query mode; type 'exit' to quit.")
    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        args.query = query
        result = run_query(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
