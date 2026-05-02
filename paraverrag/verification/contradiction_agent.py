"""Contradiction judge: incompatible claims vs reference (runs after consistency in serial mode)."""

from __future__ import annotations

from .text import truncate

KEY = "contradiction"


def build_prompt(question: str, reference: str, model_answer: str) -> str:
    q_s = truncate(question.strip(), 2000)
    ref_s = truncate(reference.strip(), 6000)
    hyp_s = truncate(model_answer.strip(), 4000)
    return (
        "You evaluate CONTRADICTION against a gold reference. Be lenient: only FAIL on strong, explicit "
        "incompatibilities.\n"
        "Given QUESTION, REFERENCE, and MODEL ANSWER: does the MODEL contradict, deny, or assert "
        "something clearly incompatible with REFERENCE?\n"
        "PASS if there is no material contradiction; minor wording differences or extra harmless detail "
        "that does not conflict with REFERENCE should PASS.\n"
        "First non-empty line must be exactly PASS or FAIL (uppercase). Optional brief reason after.\n\n"
        f"QUESTION:\n{q_s}\n\nREFERENCE:\n{ref_s}\n\nMODEL ANSWER:\n{hyp_s}\n"
    )
