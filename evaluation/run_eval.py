"""Run evaluation for ParaVerRAG and compare against baseline."""

from __future__ import annotations

import sys
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
from main import run_pipeline


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
        para = run_pipeline(ex["question"], cfg=cfg, corpus_docs=shared_corpus)
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
    results = run_eval(sample_size=50)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
