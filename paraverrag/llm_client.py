"""HTTP OpenAI-compatible LLM client (generation + judge calls)."""

from __future__ import annotations

import re
import time

import requests

from paraverrag.config import (
    LLM_JUDGE_MAX_TOKENS,
    LLM_JUDGE_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_URL,
)

_VERDICT_FIRST = re.compile(r"^(PASS|FAIL)\b", re.IGNORECASE)


def call_llm(
    user_content: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    try:
        r = requests.post(LLM_URL, json=payload, timeout=120)
    except requests.ConnectionError as e:
        raise ConnectionError(
            f"No OpenAI-compatible server at {LLM_URL!r} (connection refused or unreachable). "
            "Start your LLM server (e.g. llama.cpp `--api`, Ollama, vLLM) on that host/port, "
            "or set PARAVERAG_LLM_URL to point at it."
        ) from e
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        detail = (r.text or "")[:2000]
        raise requests.HTTPError(f"{e}\nResponse body: {detail}", response=r) from e
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def parse_first_verdict(raw: str) -> str:
    for line in raw.splitlines():
        s = line.strip().strip("*").strip()
        if not s:
            continue
        m = _VERDICT_FIRST.match(s)
        return (m.group(1).upper() if m else "") or "UNKNOWN"
    return "UNKNOWN"


def judge_llm(user: str) -> tuple[str, str, float]:
    t0 = time.perf_counter()
    raw = call_llm(
        user,
        temperature=LLM_JUDGE_TEMPERATURE,
        max_tokens=LLM_JUDGE_MAX_TOKENS,
    )
    elapsed = time.perf_counter() - t0
    return parse_first_verdict(raw), raw, elapsed
