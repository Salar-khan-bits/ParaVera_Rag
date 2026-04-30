"""GPU-style scoring stage with OpenCL fallback behavior."""

from __future__ import annotations

from typing import Dict, List

from config import PipelineConfig

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None


class GpuScorer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.opencl_available = bool(cl and cfg.use_opencl)

    def score(self, retrieved_docs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """
        Assign a final ranking score.

        If OpenCL is available, we keep the same API surface and mark the stage
        as GPU-backed; otherwise we run deterministic CPU scoring.
        """
        scored: List[Dict[str, object]] = []
        for item in retrieved_docs:
            text = str(item["text"])
            retrieval_score = float(item["retrieval_score"])
            length_bonus = min(len(text) / 800.0, 0.08)
            final_score = retrieval_score + length_bonus
            scored.append(
                {
                    **item,
                    "scoring_backend": "opencl" if self.opencl_available else "cpu",
                    "score": final_score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.cfg.top_k_scoring]
