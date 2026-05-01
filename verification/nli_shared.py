"""Shared NLI model loader for all verification agents."""

from __future__ import annotations

from typing import Any

from transformers import pipeline

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
_NLI: Any = None


def get_nli() -> Any:
    global _NLI
    if _NLI is None:
        _NLI = pipeline("text-classification", model=NLI_MODEL)
    return _NLI

