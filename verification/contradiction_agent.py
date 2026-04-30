"""Contradiction verifier: detect obvious internal conflicts."""

from __future__ import annotations

from typing import Dict


def run_contradiction_agent(answer: str) -> Dict[str, object]:
    lower = answer.lower()
    contradiction_pairs = [
        ("always", "never"),
        ("true", "false"),
        ("yes", "no"),
    ]
    contradiction_found = any(a in lower and b in lower for a, b in contradiction_pairs)
    passed = not contradiction_found
    return {
        "agent": "contradiction",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "feedback": "" if passed else "Remove contradictory claims in final response.",
    }
