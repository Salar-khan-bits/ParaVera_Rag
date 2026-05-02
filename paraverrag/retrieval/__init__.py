"""Sharded embedding retrieval: OpenCL cosine by default; optional NumPy fallback."""

from paraverrag.retrieval.sharded_retrieve import (
    load_shard_manifest,
    retrieve_from_shards,
)

__all__ = ["load_shard_manifest", "retrieve_from_shards"]
