#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: ``python -m paraverrag.evaluation`` or ``paraverrag-eval``."""

from paraverrag.evaluation import main

if __name__ == "__main__":
    main()
