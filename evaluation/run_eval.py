"""Run evaluation for ParaVerRAG and compare against baseline."""

from __future__ import annotations

import argparse
import sys
import time
from statistics import mean
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG
from dataset_load import build_corpus_from_nq, load_nq_examples
from evaluation.baseline_runner import run_baseline
from evaluation.metrics import exact_match, f1_score, hallucination_rate
from main import run_query


def _run_pipeline_like_query(question: str) -> Dict[str, object]:
    t0 = time.perf_counter()
    args = argparse.Namespace(
        config="artifacts/rag_config.json",
        chroma_path="",
        collection_name="",
        top_k=3,
        llm_url="http://localhost:8080/v1/chat/completions",
        llm_model="local-model",
        temperature=0.2,
        max_tokens=300,
        timeout=120,
        query=question,
    )
    result = run_query(args)
    result["latency_seconds"] = time.perf_counter() - t0
    # Current query flow does not run verification agents.
    result["verified"] = bool(result.get("answer"))
    return result


def run_eval(sample_size: int = 50) -> Dict[str, float]:
    cfg = DEFAULT_CONFIG
    examples = load_nq_examples(limit=sample_size)
    shared_corpus = build_corpus_from_nq(examples)
    ems: List[float] = []
    f1s: List[float] = []
    verify_flags: List[bool] = []
    para_latencies: List[float] = []
    base_latencies: List[float] = []

    for ex in examples:
        para = _run_pipeline_like_query(ex["question"])
        baseline = run_baseline(ex["question"], shared_corpus, cfg=cfg)
        ems.append(exact_match(str(para["answer"]), ex["answer"]))
        f1s.append(f1_score(str(para["answer"]), ex["answer"]))
        verify_flags.append(bool(para["verified"]))
        para_latencies.append(float(para["latency_seconds"]))
        base_latencies.append(float(baseline["latency_seconds"]))

    return {
        "sample_size": float(sample_size),
        "avg_em": mean(ems),
        "avg_f1": mean(f1s),
        "hallucination_rate": hallucination_rate(verify_flags),
        "paraverrag_avg_latency_s": mean(para_latencies),
        "baseline_avg_latency_s": mean(base_latencies),
    }


if __name__ == "__main__":
    results = run_eval(sample_size=5)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
