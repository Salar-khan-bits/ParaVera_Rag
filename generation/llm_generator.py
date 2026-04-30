"""LLM generation wrapper with local heuristic fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

import requests

from config import PipelineConfig


class LlmGenerator:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def _build_prompt(self, query: str, docs: List[Dict[str, object]], feedback: str) -> str:
        context = "\n".join(f"- {d['text']}" for d in docs)
        instruction = (
            "Answer strictly using the evidence in context. "
            "If evidence is weak, say so explicitly."
        )
        if feedback:
            instruction += f" Apply this verification feedback: {feedback}"
        return f"Question: {query}\n\nContext:\n{context}\n\nInstruction: {instruction}"

    def _fallback(self, query: str, docs: List[Dict[str, object]], feedback: str) -> str:
        top = docs[0]["text"] if docs else "No relevant context found."
        return (
            f"Draft answer for '{query}': {top}. "
            f"{'Adjusted using verifier feedback.' if feedback else ''}".strip()
        )

    def generate(
        self,
        query: str,
        docs: List[Dict[str, object]],
        feedback: str = "",
    ) -> str:
        prompt = self._build_prompt(query, docs, feedback)
        if not self.cfg.use_remote_llm_api:
            return self._fallback(query, docs, feedback)
        payload = {
            "model": self.cfg.generator_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.llm_temperature,
            "max_tokens": self.cfg.llm_max_tokens,
        }
        try:
            response = requests.post(
                self.cfg.llm_api_url,
                json=payload,
                timeout=self.cfg.llm_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback(query, docs, feedback)