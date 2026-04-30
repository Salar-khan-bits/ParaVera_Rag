"""Hallucination verifier: estimate unsupported claims."""

from __future__ import annotations

from typing import Dict, List


def run_hallucination_agent(answer: str, docs: List[Dict[str, object]]) -> Dict[str, object]:
    evidence = " ".join(str(d["text"]).lower() for d in docs)
    answer_tokens = [tok for tok in answer.lower().split() if len(tok) > 4]
    unsupported = [tok for tok in answer_tokens if tok not in evidence]
    unsupported_ratio = len(unsupported) / max(len(answer_tokens), 1)
    passed = unsupported_ratio < 0.55
    return {
        "agent": "hallucination",
        "passed": passed,
        "score": round(1.0 - unsupported_ratio, 4),
        "feedback": "" if passed else "Reduce claims not explicitly present in evidence.",
    }
