"""ParaVerRAG entrypoint implementing the full 6-stage pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import DEFAULT_CONFIG, PipelineConfig
from correction.self_corrector import run_self_correction_loop
from generation.llm_generator import LlmGenerator
from retrieval.parallel_retriever import retrieve_parallel
from scoring.gpu_scorer import GpuScorer


def _default_corpus() -> List[str]:
    return [
        "Paris is the capital city of France.",
        "Plants absorb carbon dioxide and release oxygen during photosynthesis.",
        "William Shakespeare wrote Hamlet.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Mount Everest is the highest mountain above sea level.",
        "Python is a programming language with clear syntax.",
        "The speed of light in vacuum is approximately 299,792 km/s.",
        "Albert Einstein developed the theory of relativity.",
        "Water boils at 100 C at standard atmospheric pressure.",
    ]


def run_pipeline(
    query: str,
    cfg: PipelineConfig = DEFAULT_CONFIG,
    corpus_docs: Optional[List[str]] = None,
) -> Dict[str, object]:
    start = time.perf_counter()

    # Stage 1 + 2: query encode and parallel retrieval.
    working_corpus = corpus_docs if corpus_docs else _default_corpus()
    retrieved_docs = retrieve_parallel(query, working_corpus, cfg)

    # Stage 3: GPU-style scoring.
    scored_docs = GpuScorer(cfg).score(retrieved_docs)

    # Stage 4 + 5 + correction loop.
    generator = LlmGenerator(cfg)
    corrected = run_self_correction_loop(query, scored_docs, generator, cfg)

    elapsed = time.perf_counter() - start
    return {
        "query": query,
        "answer": corrected["answer"],
        "verified": corrected["verified"],
        "attempts": corrected["attempts"],
        "verification": corrected["verification"],
        "retrieved_docs": scored_docs,
        "corpus_docs": working_corpus,
        "latency_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ParaVerRAG pipeline.")
    parser.add_argument("query", nargs="*", help="Question to answer.")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or "What is the capital of France?"
    result = run_pipeline(query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


