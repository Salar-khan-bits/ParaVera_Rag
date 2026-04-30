"""Single retrieval worker implementation (one shard)."""

from __future__ import annotations

import re
from typing import Dict, List

from retrieval.encoder import encode_text


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def search_shard(
    worker_id: int,
    query: str,
    shard_docs: List[str],
    top_k: int,
    embedding_dim: int,
) -> List[Dict[str, object]]:
    query_vec = encode_text(query, dim=embedding_dim)
    scored: List[Dict[str, object]] = []
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    for local_doc_id, doc_text in enumerate(shard_docs):
        dense_score = cosine_similarity(query_vec, encode_text(doc_text, dim=embedding_dim))
        doc_terms = set(re.findall(r"[a-z0-9]+", doc_text.lower()))
        lexical_overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        score = (0.2 * dense_score) + (0.8 * lexical_overlap)
        scored.append(
            {
                "worker_id": worker_id,
                "doc_id": f"w{worker_id}:d{local_doc_id}",
                "text": doc_text,
                "retrieval_score": score,
            }
        )

    scored.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return scored[:top_k]
