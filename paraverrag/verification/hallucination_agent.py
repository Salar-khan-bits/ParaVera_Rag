"""Hallucination judge: unsupported factual claims (runs last in serial mode)."""

from __future__ import annotations

from paraverrag.config import MAX_CHARS_PER_RETRIEVED_DOC

from .text import truncate

KEY = "hallucination"


def build_prompt(
    question: str,
    reference: str,
    model_answer: str,
    retrieved: list[str],
) -> str:
    q_s = truncate(question.strip(), 2000)
    ref_s = truncate(reference.strip(), 5000)
    hyp_s = truncate(model_answer.strip(), 4000)
    lines = ["RETRIEVED PASSAGES (supporting context for this RAG run):"]
    for i, doc in enumerate(retrieved, start=1):
        lines.append(f"{i}. {truncate(doc.strip(), MAX_CHARS_PER_RETRIEVED_DOC)}")
    ctx = "\n".join(lines)
    return (
        "You evaluate HALLUCINATION / unsupported claims. Be lenient: require clear fabrication to FAIL.\n"
        "Given QUESTION, REFERENCE, RETRIEVED PASSAGES, and MODEL ANSWER: does the MODEL introduce "
        "specific factual claims that are NOT supported by REFERENCE or the retrieved passages?\n"
        "General inference that aligns with REFERENCE or retrieval, and harmless inexact phrasing, "
        "should PASS.\n"
        "PASS if substantive claims are reasonably grounded in REFERENCE or retrieval. FAIL only for "
        "clear fabrication or claims that contradict the given sources.\n"
        "First non-empty line must be exactly PASS or FAIL (uppercase). Optional brief reason after.\n\n"
        f"{ctx}\n\nQUESTION:\n{q_s}\n\nREFERENCE:\n{ref_s}\n\nMODEL ANSWER:\n{hyp_s}\n"
    )
