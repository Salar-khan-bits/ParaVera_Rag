"""Parallel retriever that fans out shard searches to worker processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List

from config import PipelineConfig
from retrieval.worker import search_shard


def _chunk_docs(docs: List[str], shard_count: int) -> List[List[str]]:
    if not docs:
        return [[] for _ in range(shard_count)]
    shard_size = max(1, len(docs) // shard_count)
    shards = [docs[i : i + shard_size] for i in range(0, len(docs), shard_size)]
    while len(shards) < shard_count:
        shards.append([])
    return shards[:shard_count]


def retrieve_parallel(
    query: str,
    corpus_docs: Iterable[str],
    cfg: PipelineConfig,
) -> List[Dict[str, object]]:
    docs = list(corpus_docs)
    shards = _chunk_docs(docs, cfg.shard_count)
    merged_results: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=cfg.retrieval_workers) as pool:
        futures = [
            pool.submit(
                search_shard,
                worker_id,
                query,
                shard_docs,
                cfg.top_k_retrieval,
                cfg.embedding_dim,
            )
            for worker_id, shard_docs in enumerate(shards, start=1)
        ]
        for future in as_completed(futures):
            merged_results.extend(future.result())

    merged_results.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return merged_results[: cfg.top_k_retrieval]
