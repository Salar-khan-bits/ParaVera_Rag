"""Centralized configuration for the ParaVerRAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    retrieval_workers: int = 3
    shard_count: int = 3
    top_k_retrieval: int = 9
    top_k_scoring: int = 4
    max_correction_attempts: int = 3
    embedding_dim: int = 64
    use_opencl: bool = True
    generator_model_name: str = "Llama-3.2-3B"
    llm_api_url: str = "http://localhost:8080/v1/chat/completions"
    use_remote_llm_api: bool = False
    llm_timeout_seconds: float = 15.0
    llm_temperature: float = 0.1
    llm_max_tokens: int = 256
    dataset_paths: Tuple[Path, ...] = field(
        default_factory=lambda: (
            Path("data/nq"),
            Path("data/triviaqa"),
        )
    )
    logs_dir: Path = Path("logs")


DEFAULT_CONFIG = PipelineConfig()
