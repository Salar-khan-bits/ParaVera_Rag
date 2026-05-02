#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: ``python -m paraverrag.populate`` or ``paraverrag-ingest``."""

from paraverrag.populate import main

if __name__ == "__main__":
    main()
