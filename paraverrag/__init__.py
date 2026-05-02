"""ParaVerRAG: retrieval-augmented QA with Chroma, sentence-transformers, and HTTP LLM verification."""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("paraverrag")
except Exception:  # pragma: no cover - editable install without metadata
    __version__ = "0.1.0"

__all__ = ["__version__"]
