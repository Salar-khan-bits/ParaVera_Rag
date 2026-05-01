"""Consistency verifier using NLI."""

from __future__ import annotations

from typing import Dict, List, Union
from verification.nli_shared import get_nli


def run_consistency_agent(
    answer: str,
    docs: List[Union[Dict[str, object], str]],
) -> Dict[str, object]:
    if not docs:
        return {
            "agent": "consistency",
            "passed": False,
            "score": 0.0,
            "feedback": "No documents retrieved.",
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
            score = raw * 0.3
        else:
            score = 0.0
        best_score = max(best_score, score)
    passed = best_score >= 0.4
    return {
        "agent": "consistency",
        "passed": passed,
        "score": round(best_score, 4),
        "feedback": "" if passed else "Answer is inconsistent with retrieved documents.",
    }
