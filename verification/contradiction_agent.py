"""Contradiction verifier using NLI."""

from __future__ import annotations

from typing import Dict, List, Union
from verification.nli_shared import get_nli


def run_contradiction_agent(
    answer: str,
    docs: List[Union[Dict[str, object], str]],
) -> Dict[str, object]:
    if not docs:
        return {"agent": "contradiction", "passed": True, "score": 1.0, "feedback": ""}

    nli = get_nli()
    max_contradiction = 0.0
    for doc in docs:
        text = str(doc["text"]) if isinstance(doc, dict) else str(doc)
        result = nli({"text": text, "text_pair": answer})
        first = result[0] if isinstance(result, list) else result
        label = str(first["label"]).upper()
        raw = float(first["score"])
        if label == "CONTRADICTION":
            max_contradiction = max(max_contradiction, raw)

    passed = max_contradiction < 0.5
    return {
        "agent": "contradiction",
        "passed": passed,
        "score": round(1.0 - max_contradiction, 4),
        "feedback": "" if passed else "Answer contradicts retrieved documents.",
    }
