#!/usr/bin/env python3
"""Backward-compatible entrypoint. Prefer: ``python -m paraverrag.cli`` or ``paraverrag``."""

from paraverrag.cli import main

if __name__ == "__main__":
    main()
