"""Deterministic lightweight encoder used to simulate MiniLM embeddings."""

from __future__ import annotations

import hashlib
import math
from typing import List


def _hash_seed(text: str) -> bytes:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).digest()


def encode_text(text: str, dim: int = 64) -> List[float]:
    """
    Encode text into a deterministic dense vector.

    This intentionally mirrors the shape of sentence-transformer output without
    requiring heavyweight runtime dependencies.
    """
    digest = _hash_seed(text)
    vector: List[float] = []
    for i in range(dim):
        b = digest[i % len(digest)]
        value = (b / 255.0) * 2.0 - 1.0
        vector.append(value)

    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
