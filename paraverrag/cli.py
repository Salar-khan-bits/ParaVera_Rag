#!/usr/bin/env python3
"""CLI: ask a question with optional ground-truth verification."""

from __future__ import annotations

import argparse
import sys

from paraverrag.config import TOP_K
from paraverrag.rag import answer_question_simple, run_with_self_correction


def main() -> None:
    parser = argparse.ArgumentParser(description="ParaVerRAG: retrieve, generate, optional verify.")
    parser.add_argument("question", nargs="?", help="Question text")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="If set, run triple LLM judges and self-correction against this answer.",
    )
    parser.add_argument(
        "--parallel-judges",
        action="store_true",
        help="Run consistency/contradiction/hallucination judges in parallel (default: serial).",
    )
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k retrieval")
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        sys.exit(1)

    if args.ground_truth:
        ans, attempts, passed, trace, _ = run_with_self_correction(
            args.question,
            args.ground_truth,
            k=args.k,
            parallel_eval=args.parallel_judges,
        )
        print(ans)
        print(f"\n[verification passed={passed}, attempts={attempts}]", file=sys.stderr)
        for t in trace:
            print(t, file=sys.stderr)
    else:
        print(answer_question_simple(args.question, k=args.k))


if __name__ == "__main__":
    main()
