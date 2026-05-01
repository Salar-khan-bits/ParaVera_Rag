"""Hallucination verifier using NLI support."""

from __future__ import annotations

from typing import Dict, List, Union
from verification.nli_shared import get_nli


def run_hallucination_agent(
    answer: str,
    docs: List[Union[Dict[str, object], str]],
) -> Dict[str, object]:
    if not docs:
        return {
            "agent": "hallucination",
            "passed": False,
            "score": 0.0,
            "feedback": "No evidence documents provided.",
        }
    nli = get_nli()
    best_score = 0.0
    for doc in docs:
        text = str(doc["text"]) if isinstance(doc, dict) else str(doc)
        result = nli({"text": text, "text_pair": answer})
        first = result[0] if isinstance(result, list) else result
        label = str(first["label"]).upper()
        raw = float(first["score"])
        if label == "ENTAILMENT":
            score = raw
        elif label == "NEUTRAL":
            score = raw * 0.4
        else:
            score = 0.0
        best_score = max(best_score, score)
    passed = best_score >= 0.5
    return {
        "agent": "hallucination",
        "passed": passed,
        "score": round(best_score, 4),
        "feedback": "" if passed else "Reduce claims not explicitly present in evidence.",
    }
