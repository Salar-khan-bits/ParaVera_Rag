"""Evaluation metrics for answer quality and grounding."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    common = sum(min(pred_counts.get(tok, 0), gold_counts.get(tok, 0)) for tok in gold_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def hallucination_rate(verification_passes: Iterable[bool]) -> float:
    flags = list(verification_passes)
    if not flags:
        return 0.0
    fail_count = sum(1 for passed in flags if not passed)
    return fail_count / len(flags)
