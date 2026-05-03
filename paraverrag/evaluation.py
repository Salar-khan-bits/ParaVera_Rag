#!/usr/bin/env python3
"""Evaluate ParaVerRAG: JSON / Chroma / Hub test sets.

Runs two evaluation modes back-to-back by default:
  1) Serial triple-judge stage: consistency → contradiction → hallucination (separate LLM calls, in order).
  2) Parallel triple-judge stage: the same three agent prompts via a thread pool (typical wall-time win on I/O).

Aggregate verification pass/fail follows config (VERIFICATION_PASS_MODE / VERIFICATION_MIN_PASS_VOTES), not necessarily 3/3 PASS.

Per item: Query → embed → RAG → LLM answer → three LLM judges → aggregate score → self-correct if needed.
Reports NLP metrics (EM, F1, per-judge pass rates) and timing (retrieve / generate / judges wall vs serial sum).

Memory: by default the retrieval encoder is unloaded after each sample’s embed step (low peak RAM).
Use --keep-encoder-loaded to keep the encoder resident for speed. Default pipeline is ``serial`` only;
use --pipeline both for serial+parallel comparison (~2× work and higher peak usage).

Per-sample rows are written to a CSV (default ``./eval_results.csv``; see ``paraverrag.config`` / ``--results-csv``):
``query_id``, ``dataset``, ``question``, ``gold_answer``, ``predicted_answer``, ``em``, ``f1``,
``hallucination_flagged``, ``total_latency_s``, ``retrieval_latency_s``, ``scoring_latency_s`` (LLM answer generation),
``verification_latency_s`` (triple-judge wall time), ``judge_serial_sum_s`` (Σ per-judge LLM times),
``retries``, ``consistency_fired``, ``contradiction_fired``,
``hallucination_fired`` (non-PASS verdict), ``mode`` (``sequential`` / ``parallel``). Optional ``trace_json`` if enabled in config.
With ``--pipeline both``, the file contains two rows per sample (one per judge mode). The same table is printed at the end of the run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from paraverrag.config import (
    EVAL_JSON_MAX_ITEMS,
    EVAL_JSON_PATH,
    EVAL_RESULTS_CSV_INCLUDE_TRACE,
    EVAL_RESULTS_CSV_PATH,
    EVAL_SAMPLE_SOURCE,
    EVAL_SUBSET_MAX_VALID_ROWS,
    RANDOM_SEED,
    TEST_SET_SIZE,
    VERIFICATION_MIN_PASS_VOTES,
    VERIFICATION_PASS_MODE,
)
from paraverrag.data import build_eval_subset, build_eval_subset_from_chroma, load_eval_items_from_json
from paraverrag.rag import configure_memory_mode, release_peak_memory, run_with_self_correction, warmup_retrieval_stack

logger = logging.getLogger("paraverrag.eval")

_JUDGE_KEYS = ("consistency", "contradiction", "hallucination")


def _env_is_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _configure_logging(level_name: str, *, quiet_http: bool = True) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [paraverrag.eval] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    if quiet_http and level >= logging.INFO:
        for name in (
            "httpx",
            "httpcore",
            "httpcore.http11",
            "httpcore.connection",
            "huggingface_hub",
            "urllib3",
            "sentence_transformers",
            "transformers",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred: str, gold: str) -> float:
    def _tok(x: str) -> list[str]:
        x = normalize_text(x)
        x = x.translate(str.maketrans("", "", string.punctuation))
        return [t for t in x.split() if t]

    pt, gt = _tok(pred), _tok(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    c = Counter(pt) & Counter(gt)
    num_same = sum(c.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def _preview(text: str, max_len: int = 160) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _judge_pass_flags(judges_block: dict | None) -> dict[str, bool]:
    out = {k: False for k in _JUDGE_KEYS}
    if not judges_block:
        return out
    for k in _JUDGE_KEYS:
        j = judges_block.get(k) or {}
        out[k] = str(j.get("verdict", "")).upper() == "PASS"
    return out


def _judge_verdicts(judges_block: dict | None) -> dict[str, str]:
    out = {k: "" for k in _JUDGE_KEYS}
    if not judges_block:
        return out
    for k in _JUDGE_KEYS:
        j = judges_block.get(k) or {}
        out[k] = str(j.get("verdict", "")).upper()
    return out


def _judge_fired(verdict: str) -> bool:
    """True if the judge did not PASS (FAIL or UNKNOWN)."""
    return str(verdict).upper() != "PASS"


EVAL_CSV_CORE_FIELDNAMES = (
    "query_id",
    "dataset",
    "question",
    "gold_answer",
    "predicted_answer",
    "em",
    "f1",
    "hallucination_flagged",
    "total_latency_s",
    "retrieval_latency_s",
    "scoring_latency_s",
    "verification_latency_s",
    "judge_serial_sum_s",
    "retries",
    "consistency_fired",
    "contradiction_fired",
    "hallucination_fired",
    "mode",
)


def _resolve_query_id(item: dict, *, sample_index: int, dataset: str) -> str:
    raw = item.get("query_id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    safe_ds = dataset.replace("/", "_").replace(" ", "_")
    return f"{safe_ds}_{sample_index}"


def _per_sample_metrics_row(
    *,
    item: dict,
    dataset: str,
    sample_index: int,
    parallel_eval: bool,
    pred: str,
    n_att: int,
    trace: list[dict],
    timings: dict[str, float],
    step_s: float,
    em: float,
    f1: float,
    include_trace_json: bool,
) -> dict[str, str | float | bool | int]:
    """One row: user-facing metric schema (CSV + console)."""
    last = trace[-1] if trace else {}
    jf = _judge_verdicts(last.get("judges"))
    h_verdict = jf.get("hallucination", "")
    row: dict[str, str | float | bool | int] = {
        "query_id": _resolve_query_id(item, sample_index=sample_index, dataset=dataset),
        "dataset": dataset,
        "question": item["question"],
        "gold_answer": item["ground_truth"],
        "predicted_answer": pred,
        "em": em,
        "f1": f1,
        "hallucination_flagged": _judge_fired(h_verdict),
        "total_latency_s": step_s,
        "retrieval_latency_s": timings.get("retrieve_s", 0.0),
        # Time to produce the candidate answer(s) via the LLM (all attempts summed).
        "scoring_latency_s": timings.get("generate_s", 0.0),
        # Triple-judge LLM wall time (overlapped when mode=parallel).
        "verification_latency_s": timings.get("judges_s", 0.0),
        "judge_serial_sum_s": timings.get("judges_serial_sum_s", 0.0),
        "retries": max(0, n_att - 1),
        "consistency_fired": _judge_fired(jf.get("consistency", "")),
        "contradiction_fired": _judge_fired(jf.get("contradiction", "")),
        "hallucination_fired": _judge_fired(h_verdict),
        "mode": "parallel" if parallel_eval else "sequential",
    }
    if include_trace_json:
        row["trace_json"] = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    return row


def write_eval_results_csv(path: str, rows: list[dict[str, str | float | bool | int]], *, include_trace: bool) -> None:
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = list(EVAL_CSV_CORE_FIELDNAMES)
    if include_trace:
        fields.append("trace_json")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    logger.info("Wrote %d eval rows to %s", len(rows), path)


def _print_per_sample_metrics_table(rows: list[dict[str, str | float | bool | int]]) -> None:
    """Echo metric columns to stdout (full text fields; terminals may wrap)."""
    if not rows:
        return
    print(f"\n{'=' * 60}")
    print("  Per-sample metrics (same columns as CSV)")
    print(f"{'=' * 60}")
    headers = list(EVAL_CSV_CORE_FIELDNAMES)
    print(",".join(headers))
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, str) and any(c in v for c in ',"\n\r'):
                v = '"' + v.replace('"', '""') + '"'
            cells.append(str(v))
        print(",".join(cells))


@dataclass
class PipelineStats:
    label: str
    parallel_eval: bool
    n: int = 0
    ems: list[float] = field(default_factory=list)
    f1s: list[float] = field(default_factory=list)
    passes_all_judges: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    attempts: list[int] = field(default_factory=list)
    em_first: list[float] = field(default_factory=list)
    f1_first: list[float] = field(default_factory=list)
    pass_all_first: list[float] = field(default_factory=list)
    score_first: list[float] = field(default_factory=list)
    dim_pass_final: dict[str, list[float]] = field(default_factory=dict)
    dim_pass_first: dict[str, list[float]] = field(default_factory=dict)
    total_retrieve_s: float = 0.0
    total_generate_s: float = 0.0
    total_judges_wall_s: float = 0.0
    total_judges_serial_sum_s: float = 0.0
    loop_wall_s: float = 0.0

    def __post_init__(self) -> None:
        for k in _JUDGE_KEYS:
            self.dim_pass_final.setdefault(k, [])
            self.dim_pass_first.setdefault(k, [])


def _run_eval_pipeline(
    test_items: list[dict],
    *,
    label: str,
    dataset: str,
    parallel_eval: bool,
    minimal_memory: bool,
    include_trace_json: bool,
) -> tuple[PipelineStats, list[dict[str, str | float | bool | int]]]:
    stats = PipelineStats(label=label, parallel_eval=parallel_eval)
    stats.n = len(test_items)
    csv_rows: list[dict[str, str | float | bool | int]] = []
    loop_t0 = time.perf_counter()
    total = len(test_items)

    for i, item in enumerate(tqdm(test_items, desc=f"eval ({label})"), start=1):
        q, gt = item["question"], item["ground_truth"]
        logger.debug("Sample %d/%d — Q: %s", i, total, _preview(q, 200))

        step_t0 = time.perf_counter()
        pred, n_att, passed, trace, timings = run_with_self_correction(
            q, gt, parallel_eval=parallel_eval
        )
        step_s = time.perf_counter() - step_t0

        stats.total_retrieve_s += timings["retrieve_s"]
        stats.total_generate_s += timings["generate_s"]
        stats.total_judges_wall_s += timings["judges_s"]
        stats.total_judges_serial_sum_s += timings["judges_serial_sum_s"]

        em = 1.0 if exact_match(pred, gt) else 0.0
        f1 = token_f1(pred, gt)
        stats.ems.append(em)
        stats.f1s.append(f1)
        stats.passes_all_judges.append(1.0 if passed else 0.0)
        stats.attempts.append(n_att)

        last = trace[-1] if trace else {}
        stats.scores.append(float(last.get("score", 0.0)))
        j_final = last.get("judges")
        for k, ok in _judge_pass_flags(j_final).items():
            stats.dim_pass_final[k].append(1.0 if ok else 0.0)

        first = trace[0] if trace else {}
        stats.pass_all_first.append(1.0 if first.get("passed") else 0.0)
        stats.score_first.append(float(first.get("score", 0.0)))
        g0 = str(first.get("generated", ""))
        stats.em_first.append(1.0 if exact_match(g0, gt) else 0.0)
        stats.f1_first.append(token_f1(g0, gt))
        for k, ok in _judge_pass_flags(first.get("judges")).items():
            stats.dim_pass_first[k].append(1.0 if ok else 0.0)

        metrics_row = _per_sample_metrics_row(
            item=item,
            dataset=dataset,
            sample_index=i,
            parallel_eval=parallel_eval,
            pred=pred,
            n_att=n_att,
            trace=trace,
            timings=timings,
            step_s=step_s,
            em=em,
            f1=f1,
            include_trace_json=include_trace_json,
        )
        csv_rows.append(metrics_row)

        qid = metrics_row["query_id"]
        mode = metrics_row["mode"]
        logger.info(
            "Sample %d/%d [%s] query_id=%s mode=%s — total_s=%.2fs retrieval_s=%.2fs "
            "gen_s=%.2fs verify_s=%.2fs retries=%s hallu_flag=%s EM=%s F1=%.4f",
            i,
            total,
            label,
            qid,
            mode,
            step_s,
            timings.get("retrieve_s", 0.0),
            timings.get("generate_s", 0.0),
            timings.get("judges_s", 0.0),
            metrics_row["retries"],
            metrics_row["hallucination_flagged"],
            bool(em),
            f1,
        )
        if trace:
            step = trace[-1]
            logger.debug(
                "Sample %d/%d — judges: %s",
                i,
                total,
                {k: (step.get("judges") or {}).get(k, {}).get("verdict") for k in _JUDGE_KEYS},
            )

        del pred, trace, timings, last, first, g0, j_final
        if minimal_memory:
            gc.collect()
            release_peak_memory()

    stats.loop_wall_s = time.perf_counter() - loop_t0
    logger.info(
        "Pipeline %s finished: %d samples in %.2fs (avg %.2fs/sample)",
        label,
        total,
        stats.loop_wall_s,
        stats.loop_wall_s / max(total, 1),
    )
    return stats, csv_rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _print_pipeline_report(s: PipelineStats) -> None:
    n = s.n
    if n == 0:
        return
    print(f"\n{'=' * 60}")
    print(f"  Pipeline: {s.label}  (parallel_eval={s.parallel_eval})")
    print(f"{'=' * 60}")
    print(
        f"Verification rule: {VERIFICATION_PASS_MODE} "
        f"(min PASS votes={VERIFICATION_MIN_PASS_VOTES} when lenient)"
    )
    print("\n--- After self-correction (final answer) ---")
    print(f"Exact match (normalized):     {_mean(s.ems):.4f}")
    print(f"Token F1 (mean):              {_mean(s.f1s):.4f}")
    print(f"Verification pass (aggregate): {_mean(s.passes_all_judges):.4f}")
    print(f"Judge score mean (0–1):       {_mean(s.scores):.4f}")
    print(f"Average attempts:             {_mean([float(x) for x in s.attempts]):.4f}")
    for k in _JUDGE_KEYS:
        print(f"  {k + ' PASS rate:':<28}{_mean(s.dim_pass_final[k]):.4f}")

    print("\n--- First attempt only ---")
    print(f"Exact match (normalized):     {_mean(s.em_first):.4f}")
    print(f"Token F1 (mean):              {_mean(s.f1_first):.4f}")
    print(f"Verification pass (aggregate): {_mean(s.pass_all_first):.4f}")
    print(f"Judge score mean (0–1):       {_mean(s.score_first):.4f}")
    for k in _JUDGE_KEYS:
        print(f"  {k + ' PASS rate:':<28}{_mean(s.dim_pass_first[k]):.4f}")

    print("\n--- Time / efficiency (sum over all samples) ---")
    print(f"Retrieve (embed + Chroma):    {s.total_retrieve_s:.2f}s")
    print(f"LLM generation:             {s.total_generate_s:.2f}s")
    print(f"Triple judges (wall clock): {s.total_judges_wall_s:.2f}s")
    print(f"Triple judges (Σ per-call): {s.total_judges_serial_sum_s:.2f}s  (ideal serial sum of LLM latencies)")
    if s.parallel_eval and s.total_judges_serial_sum_s > 0:
        saved = max(0.0, s.total_judges_serial_sum_s - s.total_judges_wall_s)
        pct = 100.0 * saved / s.total_judges_serial_sum_s if s.total_judges_serial_sum_s else 0.0
        print(f"Parallel wall vs Σ calls:   saved ~{saved:.2f}s (~{pct:.1f}% of summed judge latency)")
    print(f"End-to-end loop wall:         {s.loop_wall_s:.2f}s")
    print(f"Per-sample wall (avg):        {s.loop_wall_s / n:.2f}s")


def _print_comparison(serial: PipelineStats, parallel: PipelineStats) -> None:
    if serial.n == 0 or parallel.n == 0:
        return
    print(f"\n{'=' * 60}")
    print("  Comparison: serial vs parallel judge stage")
    print(f"{'=' * 60}")
    print(
        f"End-to-end wall:  serial {serial.loop_wall_s:.2f}s  |  parallel {parallel.loop_wall_s:.2f}s  "
        f"|  Δ {parallel.loop_wall_s - serial.loop_wall_s:+.2f}s"
    )
    print(
        f"Judge wall total: serial {serial.total_judges_wall_s:.2f}s  |  parallel {parallel.total_judges_wall_s:.2f}s  "
        f"|  Δ {parallel.total_judges_wall_s - serial.total_judges_wall_s:+.2f}s"
    )
    print(
        f"Verification pass: serial {_mean(serial.passes_all_judges):.4f}  |  parallel {_mean(parallel.passes_all_judges):.4f}"
    )
    print(
        f"EM (final):       serial {_mean(serial.ems):.4f}  |  parallel {_mean(parallel.ems):.4f}"
    )


def main() -> None:
    env_level = os.environ.get("PARAVERAG_EVAL_LOG_LEVEL", "").strip().upper()
    parser = argparse.ArgumentParser(
        description="Evaluate ParaVerRAG (minimal RAM by default; single pipeline unless --pipeline both)."
    )
    parser.add_argument(
        "--log-level",
        default=env_level if env_level in ("DEBUG", "INFO", "WARNING", "ERROR") else "INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO, or PARAVERAG_EVAL_LOG_LEVEL).",
    )
    parser.add_argument("--debug", action="store_true", help="Shorthand for --log-level DEBUG.")
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Use EVAL_SAMPLE_SOURCE=hub (HF parquet), ignoring default json.",
    )
    parser.add_argument(
        "--from-chroma",
        action="store_true",
        help="Use EVAL_SAMPLE_SOURCE=chroma (sample from vector DB).",
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        default=None,
        help=f"Path to eval JSON (default: {EVAL_JSON_PATH}).",
    )
    parser.add_argument(
        "--keep-encoder-loaded",
        action="store_true",
        help="Keep the SentenceTransformer encoder in memory across samples (faster, higher RAM/VRAM). "
        "Default is to unload after each sample’s retrieval when using minimal memory.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Force minimal-memory mode even if --keep-encoder-loaded was set (encoder unloads after each embed).",
    )
    parser.add_argument(
        "--pipeline",
        choices=("both", "serial", "parallel"),
        default="serial",
        help="Run triple judges serially, in parallel, or both (default: serial — use both only for comparison).",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Write one CSV row per eval sample (and per pipeline when --pipeline both). "
        f"Default: config EVAL_RESULTS_CSV_PATH ({EVAL_RESULTS_CSV_PATH!r}).",
    )
    parser.add_argument(
        "--no-results-csv",
        action="store_true",
        help="Disable writing the per-sample results CSV.",
    )
    args = parser.parse_args()
    level_name = "DEBUG" if args.debug else args.log_level
    _configure_logging(level_name, quiet_http=not args.debug)

    force_low = args.low_memory or _env_is_truthy("PARAVERAG_EVAL_LOW_MEMORY")
    keep_loaded = args.keep_encoder_loaded or _env_is_truthy("PARAVERAG_EVAL_KEEP_ENCODER_LOADED")
    # Default: unload encoder after each retrieval (minimal peak memory). Opt out with --keep-encoder-loaded.
    serialize_models = force_low or not keep_loaded

    if args.from_hub and args.from_chroma:
        parser.error("Use only one of --from-hub and --from-chroma.")
    if args.from_hub:
        sample_source = "hub"
    elif args.from_chroma:
        sample_source = "chroma"
    else:
        sample_source = EVAL_SAMPLE_SOURCE
    if sample_source not in ("json", "chroma", "hub"):
        parser.error(f"Invalid EVAL_SAMPLE_SOURCE / PARAVERAG_EVAL_SAMPLE_SOURCE: {sample_source!r}")

    logger.info(
        "Starting evaluation: sample_source=%s RANDOM_SEED=%s TEST_SET_SIZE=%s pipeline=%s",
        sample_source,
        RANDOM_SEED,
        TEST_SET_SIZE,
        args.pipeline,
    )
    if sample_source == "json":
        logger.info(
            "JSON eval: row cap from EVAL_JSON_MAX_ITEMS / env PARAVERAG_EVAL_JSON_MAX_ITEMS / EVAL_JSON_MAX_ITEMS.",
        )
    elif sample_source == "hub":
        logger.info(
            "Hub eval: reservoir size=TEST_SET_SIZE; parquet scan cap=EVAL_SUBSET_MAX_VALID_ROWS=%s",
            EVAL_SUBSET_MAX_VALID_ROWS,
        )
    eval_json_path = args.eval_json or EVAL_JSON_PATH
    if sample_source == "json":
        if EVAL_JSON_MAX_ITEMS is None:
            logger.info("Eval JSON path: %s (using all rows in file)", eval_json_path)
        else:
            logger.info(
                "Eval JSON path: %s (subsample at most %s rows)",
                eval_json_path,
                EVAL_JSON_MAX_ITEMS,
            )
    if serialize_models:
        logger.info(
            "Minimal-memory eval: encoder unloads after each sample’s retrieval (default). "
            "Use --keep-encoder-loaded or PARAVERAG_EVAL_KEEP_ENCODER_LOADED=1 to keep it resident for speed."
        )
    else:
        logger.info(
            "Encoder stays loaded across eval items (--keep-encoder-loaded). Higher peak RAM/VRAM; faster retrieval."
        )

    t0 = time.perf_counter()
    configure_memory_mode(serialize_models=serialize_models)
    if sample_source == "json":
        test_items = load_eval_items_from_json(
            eval_json_path,
            max_items=EVAL_JSON_MAX_ITEMS,
            seed=RANDOM_SEED,
        )
    elif sample_source == "chroma":
        logger.info("Building eval subset from Chroma at ./chroma_db (no dataset download).")
        test_items = build_eval_subset_from_chroma(TEST_SET_SIZE, RANDOM_SEED)
    else:
        test_items = build_eval_subset(
            TEST_SET_SIZE, RANDOM_SEED, max_valid_rows=EVAL_SUBSET_MAX_VALID_ROWS
        )
    subset_s = time.perf_counter() - t0
    logger.info("Built eval subset: %d items in %.2fs", len(test_items), subset_s)

    if sample_source == "hub" and len(test_items) < TEST_SET_SIZE and TEST_SET_SIZE > 0:
        logger.warning(
            "Only %d eval samples (requested %d); increase EVAL_SUBSET_MAX_VALID_ROWS or set to None.",
            len(test_items),
            TEST_SET_SIZE,
        )
        print(
            f"Note: only {len(test_items)} eval samples (requested {TEST_SET_SIZE}); "
            "increase EVAL_SUBSET_MAX_VALID_ROWS in config or set to None for a full scan.",
            flush=True,
        )
    print(
        f"Evaluating on {len(test_items)} samples (seed={RANDOM_SEED}); "
        f"pipeline mode={args.pipeline} (serial judges vs parallel judges).",
        flush=True,
    )
    if not test_items:
        logger.error("No eval samples; exiting.")
        print("No eval samples; exiting.")
        return

    if args.no_results_csv:
        results_csv_path: str | None = None
    elif args.results_csv is not None:
        results_csv_path = args.results_csv.strip() or None
    else:
        results_csv_path = EVAL_RESULTS_CSV_PATH

    stats_serial: PipelineStats | None = None
    stats_parallel: PipelineStats | None = None
    all_csv_rows: list[dict[str, str | float | bool | int]] = []

    minimal_mem = serialize_models
    trace_in_csv = EVAL_RESULTS_CSV_INCLUDE_TRACE

    # Pay encoder + Chroma cold start once so sample 1 total_latency_s / retrieval_latency_s are comparable.
    warmup_retrieval_stack()

    if sample_source == "json":
        dataset_label = f"json:{Path(eval_json_path).name}"
    elif sample_source == "chroma":
        dataset_label = "chroma"
    else:
        dataset_label = "hub"

    if args.pipeline in ("both", "serial"):
        stats_serial, rows_s = _run_eval_pipeline(
            test_items,
            label="serial_judges",
            dataset=dataset_label,
            parallel_eval=False,
            minimal_memory=minimal_mem,
            include_trace_json=trace_in_csv,
        )
        all_csv_rows.extend(rows_s)
        _print_pipeline_report(stats_serial)
        if args.pipeline == "both":
            gc.collect()
            release_peak_memory()

    if args.pipeline in ("both", "parallel"):
        stats_parallel, rows_p = _run_eval_pipeline(
            test_items,
            label="parallel_judges",
            dataset=dataset_label,
            parallel_eval=True,
            minimal_memory=minimal_mem,
            include_trace_json=trace_in_csv,
        )
        all_csv_rows.extend(rows_p)
        _print_pipeline_report(stats_parallel)

    if results_csv_path and all_csv_rows:
        write_eval_results_csv(
            results_csv_path, all_csv_rows, include_trace=trace_in_csv
        )
        print(f"Per-sample eval CSV: {results_csv_path} ({len(all_csv_rows)} rows)", flush=True)

    if all_csv_rows:
        _print_per_sample_metrics_table(all_csv_rows)
    elif results_csv_path and not all_csv_rows:
        logger.warning("Results CSV path set but no rows to write.")

    if stats_serial is not None and stats_parallel is not None:
        _print_comparison(stats_serial, stats_parallel)


if __name__ == "__main__":
    main()
