"""Run consistency → contradiction → hallucination judges serially or in parallel."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from paraverrag.config import (
    CPU_MAX_THREADS,
    EVAL_PARALLEL_JUDGE_WORKERS,
    VERIFICATION_MIN_PASS_VOTES,
    VERIFICATION_PASS_MODE,
)
from paraverrag.llm_client import judge_llm

from . import consistency_agent, contradiction_agent, hallucination_agent

_JUDGE_KEYS = (
    consistency_agent.KEY,
    contradiction_agent.KEY,
    hallucination_agent.KEY,
)


def _build_jobs(
    question: str,
    reference: str,
    model_answer: str,
    retrieved: list[str],
) -> list[tuple[str, str]]:
    return [
        (
            consistency_agent.KEY,
            consistency_agent.build_prompt(question, reference, model_answer),
        ),
        (
            contradiction_agent.KEY,
            contradiction_agent.build_prompt(question, reference, model_answer),
        ),
        (
            hallucination_agent.KEY,
            hallucination_agent.build_prompt(question, reference, model_answer, retrieved),
        ),
    ]


def _aggregate_pass(n_pass: int, n_total: int) -> bool:
    if VERIFICATION_PASS_MODE == "strict":
        return n_pass == n_total
    need = max(1, min(n_total, VERIFICATION_MIN_PASS_VOTES))
    return n_pass >= need


def evaluate_answer_triple_judges(
    question: str,
    reference: str,
    model_answer: str,
    retrieved: list[str],
    *,
    parallel: bool,
    max_workers: int = EVAL_PARALLEL_JUDGE_WORKERS,
) -> tuple[bool, float, dict[str, Any]]:
    """Run the three verification agents; serial order is consistency → contradiction → hallucination.

    When parallel=True, all three LLM calls run concurrently (same prompts, faster wall clock).

    Returns (aggregate_passed, score_mean_0_to_1, details).
    """
    jobs = _build_jobs(question, reference, model_answer, retrieved)
    judges: dict[str, dict[str, Any]] = {}
    pool_workers = max(1, min(max_workers, len(jobs), CPU_MAX_THREADS))

    if parallel:
        t_wall0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            futs = {ex.submit(judge_llm, prompt): name for name, prompt in jobs}
            for fut in as_completed(futs):
                name = futs[fut]
                verdict, raw, elapsed = fut.result()
                judges[name] = {
                    "verdict": verdict,
                    "elapsed_s": elapsed,
                    "raw": raw[:120],
                }
        wall_s = time.perf_counter() - t_wall0
        serial_sum_s = sum(j["elapsed_s"] for j in judges.values())
    else:
        wall_s = 0.0
        serial_sum_s = 0.0
        for name, prompt in jobs:
            verdict, raw, elapsed = judge_llm(prompt)
            wall_s += elapsed
            serial_sum_s += elapsed
            judges[name] = {
                "verdict": verdict,
                "elapsed_s": elapsed,
                "raw": raw[:120],
            }

    verdicts = [judges[k]["verdict"] for k in _JUDGE_KEYS]
    oks = [v == "PASS" for v in verdicts]
    n_pass = sum(1 for ok in oks if ok)
    score = n_pass / 3.0
    all_passed = _aggregate_pass(n_pass, len(_JUDGE_KEYS))

    details: dict[str, Any] = {
        "judges": judges,
        "judge_mode": "parallel" if parallel else "serial",
        "score": score,
        "judge_wall_s": wall_s if parallel else serial_sum_s,
        "judge_serial_sum_s": serial_sum_s,
    }
    return all_passed, score, details
