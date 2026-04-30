"""Consistency verifier: does answer align with retrieved evidence?"""

from __future__ import annotations

from typing import Dict, List


def run_consistency_agent(answer: str, docs: List[Dict[str, object]]) -> Dict[str, object]:
    answer_words = set(answer.lower().split())
    evidence_words = set()
    for doc in docs:
        evidence_words.update(str(doc["text"]).lower().split())
    overlap = len(answer_words & evidence_words)
    ratio = overlap / max(len(answer_words), 1)
    passed = ratio >= 0.25
    return {
        "agent": "consistency",
        "passed": passed,
        "score": round(ratio, 4),
        "feedback": "" if passed else "Ground more phrases directly in retrieved context.",
    }
