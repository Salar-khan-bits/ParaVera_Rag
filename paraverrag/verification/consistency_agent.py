"""Consistency judge: model answer vs gold reference (runs first in serial mode)."""

from __future__ import annotations

from .text import truncate

KEY = "consistency"


def build_prompt(question: str, reference: str, model_answer: str) -> str:
    q_s = truncate(question.strip(), 2000)
    ref_s = truncate(reference.strip(), 6000)
    hyp_s = truncate(model_answer.strip(), 4000)
    return (
        "You evaluate CONSISTENCY with a gold reference. Be lenient: prefer PASS when the model is "
        "directionally right.\n"
        "Given QUESTION, REFERENCE (authoritative), and MODEL ANSWER: does the MODEL convey the same "
        "core factual content as REFERENCE for this question? Paraphrases, partial answers, and minor "
        "omissions are acceptable. Only FAIL on clear wrong facts, missing the main point, or evasion "
        '(e.g. "I don\'t know") when REFERENCE answers clearly.\n'
        "First non-empty line must be exactly PASS or FAIL (uppercase). Optional brief reason after.\n\n"
        f"QUESTION:\n{q_s}\n\nREFERENCE:\n{ref_s}\n\nMODEL ANSWER:\n{hyp_s}\n"
    )
