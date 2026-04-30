"""Sequential baseline for latency comparison against ParaVerRAG."""

from __future__ import annotations

import time
from typing import Dict, List

from config import PipelineConfig
from generation.llm_generator import LlmGenerator
from retrieval.worker import search_shard
from scoring.gpu_scorer import GpuScorer


def run_baseline(query: str, corpus_docs: List[str], cfg: PipelineConfig) -> Dict[str, object]:
    t0 = time.perf_counter()
    retrieved = search_shard(
        worker_id=0,
        query=query,
        shard_docs=corpus_docs,
        top_k=cfg.top_k_retrieval,
        embedding_dim=cfg.embedding_dim,
    )
    scored = GpuScorer(cfg).score(retrieved)
    answer = LlmGenerator(cfg).generate(query, scored, feedback="")
    elapsed = time.perf_counter() - t0
    return {"answer": answer, "latency_seconds": elapsed}
