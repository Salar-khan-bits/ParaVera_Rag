"""Self-correction loop driven by verification feedback."""

from __future__ import annotations

from typing import Dict, List

from config import PipelineConfig
from generation.llm_generator import LlmGenerator
from verification.agent_runner import run_verification_agents


def run_self_correction_loop(
    query: str,
    scored_docs: List[Dict[str, object]],
    generator: LlmGenerator,
    cfg: PipelineConfig,
) -> Dict[str, object]:
    attempt = 0
    feedback = ""
    final_answer = ""
    final_verification: Dict[str, object] = {"all_passed": False, "results": [], "feedback": ""}

    while attempt < cfg.max_correction_attempts:
        attempt += 1
        final_answer = generator.generate(query, scored_docs, feedback=feedback)
        final_verification = run_verification_agents(final_answer, scored_docs)
        if final_verification["all_passed"]:
            break
        feedback = str(final_verification["feedback"])

    return {
        "answer": final_answer,
        "attempts": attempt,
        "verified": bool(final_verification["all_passed"]),
        "verification": final_verification,
    }
