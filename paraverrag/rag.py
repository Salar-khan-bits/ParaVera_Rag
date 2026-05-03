#!/usr/bin/env python3
"""RAG pipeline: Chroma retrieval, HTTP LLM generation, triple LLM judges, optional self-correction."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from paraverrag.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    ENCODER_MODEL,
    LLM_MAX_TOKENS,
    LLM_MAX_USER_CHARS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_URL,
    MAX_CHARS_PER_RETRIEVED_DOC,
    MAX_CORRECTION_ATTEMPTS,
    RETRIEVAL_BACKEND,
    TORCH_DEVICE,
    TOP_K,
)
from paraverrag.llm_client import call_llm
from paraverrag.verification import evaluate_answer_triple_judges

_encoder: SentenceTransformer | None = None
_chroma_collection: Any | None = None
_serialize_models: bool = False
_runtime_logged: bool = False


def _torch_encoder_device() -> str:
    """SentenceTransformer device: cuda or cpu."""
    mode = TORCH_DEVICE
    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if mode in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TORCH_DEVICE / PARAVERAG_TORCH_DEVICE requests CUDA but "
                "torch.cuda.is_available() is False. Install a CUDA build of PyTorch "
                "(https://pytorch.org) and ensure the GPU is visible (try `nvidia-smi`). "
                "On WSL2 you need NVIDIA drivers on Windows and a matching torch+cuda wheel."
            )
        return "cuda"
    if mode == "cpu":
        return "cpu"
    raise ValueError(f"Unknown TORCH_DEVICE / PARAVERAG_TORCH_DEVICE: {mode!r} (use auto, cuda, cpu)")


def _log_runtime_once() -> None:
    global _runtime_logged
    if _runtime_logged:
        return
    _runtime_logged = True
    enc_dev = _torch_encoder_device()
    print(
        f"[paraverrag] torch {torch.__version__} | cuda.is_available()={torch.cuda.is_available()}",
        file=sys.stderr,
    )
    if torch.cuda.is_available():
        print(f"[paraverrag] GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"[paraverrag] encoder uses: {enc_dev}", file=sys.stderr)
    print(
        "[paraverrag] generation + triple judges use HTTP LLM at "
        f"{LLM_URL} — that server must be running; GPU there depends on the server, not this script.",
        file=sys.stderr,
    )


def configure_memory_mode(*, serialize_models: bool = False) -> None:
    """When True, drop heavy models between pipeline stages to cap peak RAM (slower)."""
    global _serialize_models
    _serialize_models = serialize_models


def release_peak_memory() -> None:
    """Best-effort drop of allocator/GPU cache pressure between eval samples."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_encoder() -> None:
    global _encoder
    if _encoder is not None:
        del _encoder
        _encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _log_runtime_once()
        enc_dev = _torch_encoder_device()
        if enc_dev == "cuda":
            torch.backends.cudnn.benchmark = True
        _encoder = SentenceTransformer(ENCODER_MODEL, device=enc_dev)
    return _encoder


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def warmup_retrieval_stack() -> None:
    """Load encoder + Chroma client and run a tiny encode so later timed retrieval excludes cold start."""
    _log_runtime_once()
    model = get_encoder()
    model.encode(["."], convert_to_numpy=True, show_progress_bar=False)
    if RETRIEVAL_BACKEND == "sharded":
        from paraverrag.retrieval.sharded_retrieve import load_shard_manifest

        load_shard_manifest()
    else:
        get_collection()


def retrieve_answers(question: str, k: int = TOP_K) -> list[str]:
    model = get_encoder()
    q_emb = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    q_np = np.asarray(q_emb, dtype=np.float32, order="C")[0]

    if RETRIEVAL_BACKEND == "sharded":
        from paraverrag.retrieval.sharded_retrieve import retrieve_from_shards

        out = retrieve_from_shards(q_np, k)
    else:
        col = get_collection()
        res = col.query(query_embeddings=q_emb.tolist(), n_results=k, include=["documents"])
        docs = res.get("documents") or [[]]
        out = list(docs[0]) if docs and docs[0] else []

    if _serialize_models:
        _unload_encoder()
    return out


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n... [truncated]"


def build_user_prompt(
    question: str,
    retrieved: list[str],
    extra_feedback: list[str] | None = None,
) -> str:
    lines = [
        "You are a careful assistant. Use ONLY the numbered excerpts below as your source of facts.",
        "",
        "How to answer:",
        "- Give a direct answer in your own words. Do not write meta phrases like \"According to answer 1\" or "
        "\"Based on passage 2\"; synthesize across excerpts naturally.",
        "- If the excerpts contain information that helps answer the question (fully or partly), state it. Combine "
        "details from multiple excerpts when needed.",
        '- Reserve "I don\'t know" for when excerpts are missing, empty, or clearly unrelated with no usable facts. '
        "Do not use it when excerpts are on-topic but imperfect—then give the best supported summary.",
        '- For questions that ask what something "is called" or for its name, prefer the exact term or title that '
        "appears in the excerpts; do not replace it with a different but related idea (e.g. a sibling concept).",
        "- Be concise but include the main factual points the excerpts support (lists, dates, names) when relevant.",
        "",
        "Excerpts:",
    ]
    for i, ans in enumerate(retrieved, start=1):
        lines.append(f"{i}. {_truncate(ans, MAX_CHARS_PER_RETRIEVED_DOC)}")
    if extra_feedback:
        lines.append("")
        lines.extend(extra_feedback)
    lines.extend(["", f"Question: {question}", "", "Answer:"])
    out = "\n".join(lines)
    return _truncate(out, LLM_MAX_USER_CHARS)


def run_with_self_correction(
    question: str,
    ground_truth: str,
    k: int = TOP_K,
    max_attempts: int = MAX_CORRECTION_ATTEMPTS,
    *,
    parallel_eval: bool = False,
) -> tuple[str, int, bool, list[dict[str, Any]], dict[str, float]]:
    """RAG + generate + triple judges + self-correction.

    parallel_eval: if True, the three judge LLM calls run concurrently (I/O bound); if False, serial.

    Returns (final_answer, attempts_used, verification_passed, trace, timing_totals).
    """
    feedback_lines: list[str] = []
    verify_trace: list[dict[str, Any]] = []
    timing_totals: dict[str, float] = {
        "retrieve_s": 0.0,
        "generate_s": 0.0,
        "judges_s": 0.0,
        "judges_serial_sum_s": 0.0,
    }

    # One-time model / DB connection cost must not be charged to retrieval_latency_s.
    get_encoder()
    get_collection()
    t_r0 = time.perf_counter()
    retrieved = retrieve_answers(question, k=k)
    timing_totals["retrieve_s"] += time.perf_counter() - t_r0

    generated = ""
    for attempt in range(1, max_attempts + 1):
        t_g0 = time.perf_counter()
        prompt = build_user_prompt(question, retrieved, extra_feedback=feedback_lines or None)
        generated = call_llm(prompt)
        timing_totals["generate_s"] += time.perf_counter() - t_g0

        ok, score, jdetails = evaluate_answer_triple_judges(
            question,
            ground_truth,
            generated,
            retrieved,
            parallel=parallel_eval,
        )
        timing_totals["judges_s"] += jdetails["judge_wall_s"]
        timing_totals["judges_serial_sum_s"] += jdetails["judge_serial_sum_s"]

        row: dict[str, Any] = {
            "attempt": attempt,
            "generated": generated,
            "passed": ok,
            "score": score,
            "parallel_eval": parallel_eval,
            **jdetails,
        }
        verify_trace.append(row)
        if ok:
            if _serialize_models:
                release_peak_memory()
            return generated, attempt, True, verify_trace, timing_totals

        feedback_lines = []
        for judge_name, judge_details in jdetails.get("judges", {}).items():
            if judge_details.get("verdict") == "FAIL":
                reason = judge_details.get("raw", "")[:200]
                feedback_lines.append(
                    f"The {judge_name} check failed. Reason: {reason}"
                )
        if not feedback_lines:
            feedback_lines = ["Please review your answer for accuracy."]
        if _serialize_models:
            release_peak_memory()

    if _serialize_models:
        release_peak_memory()
    return generated, max_attempts, False, verify_trace, timing_totals


def answer_question_simple(question: str, k: int = TOP_K) -> str:
    """Retrieve + one LLM call (no ground-truth verification)."""
    retrieved = retrieve_answers(question, k=k)
    prompt = build_user_prompt(question, retrieved)
    return call_llm(prompt)


