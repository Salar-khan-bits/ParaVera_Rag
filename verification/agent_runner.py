"""Parallel verification runner for the three verifier agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from verification.consistency_agent import run_consistency_agent
from verification.contradiction_agent import run_contradiction_agent
from verification.hallucination_agent import run_hallucination_agent


def run_verification_agents(
    answer: str, docs: List[Dict[str, object]]
) -> Dict[str, object]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_consistency = pool.submit(run_consistency_agent, answer, docs)
        future_contradiction = pool.submit(run_contradiction_agent, answer, docs)
        future_hallucination = pool.submit(run_hallucination_agent, answer, docs)
        agent_results = [
            future_consistency.result(),
            future_contradiction.result(),
            future_hallucination.result(),
        ]

    all_passed = all(bool(result["passed"]) for result in agent_results)
    feedback = " ".join(r["feedback"] for r in agent_results if r["feedback"]).strip()
    return {
        "all_passed": all_passed,
        "results": agent_results,
        "feedback": feedback,
    }
