"""
Interactive RAG CLI using local llama.cpp-compatible server.

Expected default files (created by embeddings.py):
  artifacts/chroma_db/
  artifacts/rag_config.json

Usage:
  python rag_cli.py
  python rag_cli.py --top-k 5 --llm-url http://localhost:8080/v1/chat/completions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
import requests
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG CLI")
    parser.add_argument("--config", default="artifacts/rag_config.json", help="RAG config path")
    parser.add_argument(
        "--chroma-path",
        default="",
        help="Override ChromaDB path (optional, uses config otherwise)",
    )
    parser.add_argument("--collection-name", default="", help="Override Chroma collection name")
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8080/v1/chat/completions",
        help="Local llama server OpenAI-compatible endpoint",
    )
    parser.add_argument("--llm-model", default="local-model", help="Model name sent to server")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=300, help="LLM max tokens")
    parser.add_argument("--timeout", type=int, default=120, help="LLM request timeout (seconds)")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. Run `python embeddings.py` first."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    collection: Any,
    top_k: int,
) -> list[tuple[dict[str, str], float]]:
    query_vec = (
        embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    )
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[tuple[dict[str, str], float]] = []
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metadatas, distances):
        score = 1.0 - float(dist)
        row = {"query": str((meta or {}).get("query", "")), "answer": str(doc)}
        hits.append((row, score))
    return hits


def build_prompt(query: str, hits: list[tuple[dict[str, str], float]]) -> str:
    context_lines = []
    for i, (row, score) in enumerate(hits, start=1):
        context_lines.append(
            f"[{i}] score={score:.4f}\nQuestion: {row['query']}\nAnswer: {row['answer']}"
        )
    context = "\n\n".join(context_lines)
    return (
        "You are a factual QA assistant. Use only the provided context.\n"
        "If the answer is not clearly present in context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer:"
    )


def call_llm(
    llm_url: str,
    llm_model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(llm_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    chroma_path = Path(args.chroma_path or cfg["chroma_path"])
    collection_name = args.collection_name or cfg["collection_name"]
    embed_model_name = str(cfg["embed_model"])

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found: {chroma_path}. Run `python embeddings.py` first."
        )

    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)
    print(f"Loading ChromaDB: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    print(f"Ready. Collection `{collection_name}` docs: {collection.count():,}")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        hits = retrieve(query, embedder, collection, args.top_k)
        if not hits:
            print("No retrieval hits found.\n")
            continue

        print("\nRetrieved context:")
        for i, (row, score) in enumerate(hits, start=1):
            preview = row["answer"].replace("\n", " ")[:140]
            print(f"  [{i}] score={score:.4f} | {preview}...")

        prompt = build_prompt(query, hits)
        try:
            answer = call_llm(
                llm_url=args.llm_url,
                llm_model=args.llm_model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"\nLLM request failed: {exc}\n")
            continue

        print("\nAnswer:")
        print(answer)
        print()


if __name__ == "__main__":
    main()

