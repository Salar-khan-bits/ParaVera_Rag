"""Central configuration for ParaVerRAG MVP."""

import os

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "answers_collection"
ENCODER_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Global cap on Python thread pools (logical CPUs). Env: PARAVERAG_CPU_MAX_THREADS (default: os.cpu_count()).
_detected_cpus = os.cpu_count()
_DEFAULT_LOGICAL_CPUS = max(1, _detected_cpus if _detected_cpus is not None else 4)
_cpu_max_env = os.environ.get("PARAVERAG_CPU_MAX_THREADS", "").strip()
if _cpu_max_env:
    CPU_MAX_THREADS = max(1, min(_DEFAULT_LOGICAL_CPUS, int(_cpu_max_env)))
else:
    CPU_MAX_THREADS = _DEFAULT_LOGICAL_CPUS

# Retrieval: "chroma" (default HNSW) | "sharded" (NPZ shards; cosine on shards uses OpenCL by default).
RETRIEVAL_BACKEND = os.environ.get("PARAVERAG_RETRIEVAL_BACKEND", "chroma").strip().lower()
if RETRIEVAL_BACKEND not in ("chroma", "sharded"):
    RETRIEVAL_BACKEND = "chroma"
RETRIEVAL_SHARD_DIR = os.environ.get("PARAVERAG_RETRIEVAL_SHARD_DIR", "./retrieval_shards").strip()
_RETRIEVAL_SHARD_COUNT_ENV = os.environ.get("PARAVERAG_RETRIEVAL_SHARD_COUNT", "").strip()
RETRIEVAL_SHARD_COUNT = max(1, int(_RETRIEVAL_SHARD_COUNT_ENV)) if _RETRIEVAL_SHARD_COUNT_ENV else 4
_RETRIEVAL_WORKERS_ENV = os.environ.get("PARAVERAG_RETRIEVAL_CPU_WORKERS", "").strip()
if _RETRIEVAL_WORKERS_ENV:
    _rw = max(1, int(_RETRIEVAL_WORKERS_ENV))
    RETRIEVAL_CPU_WORKERS = max(1, min(CPU_MAX_THREADS, RETRIEVAL_SHARD_COUNT, _rw))
else:
    RETRIEVAL_CPU_WORKERS = max(1, min(CPU_MAX_THREADS, RETRIEVAL_SHARD_COUNT))
# Sharded mode: cosine similarity vs shard rows runs in OpenCL (see opencl_cosine.py). NumPy fallback only if:
#   PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1
_RETRIEVAL_NUMPY_COS = os.environ.get("PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE", "").strip().lower()
RETRIEVAL_ALLOW_NUMPY_COSINE = _RETRIEVAL_NUMPY_COS in ("1", "true", "yes", "on")
# OpenCL device for sharded cosine (PyOpenCL): cpu | gpu | auto. Default **cpu** (e.g. POCL); use **auto** if you
# only have a GPU ICD (e.g. nvidia-opencl-icd) and no CPU OpenCL platform.
_OPENCL_DEV = os.environ.get("PARAVERAG_OPENCL_DEVICE", "cpu").strip().lower()
if _OPENCL_DEV not in ("auto", "cpu", "gpu"):
    OPENCL_DEVICE = "cpu"
else:
    OPENCL_DEVICE = _OPENCL_DEV
# When ingesting, also write shard npz files under RETRIEVAL_SHARD_DIR (required for RETRIEVAL_BACKEND=sharded).
_pop_shards_raw = os.environ.get("PARAVERAG_POPULATE_WRITE_RETRIEVAL_SHARDS", "").strip().lower()
if _pop_shards_raw in ("0", "false", "no", "off"):
    POPULATE_WRITE_RETRIEVAL_SHARDS = False
elif _pop_shards_raw in ("1", "true", "yes", "on"):
    POPULATE_WRITE_RETRIEVAL_SHARDS = True
else:
    POPULATE_WRITE_RETRIEVAL_SHARDS = True

# PyTorch device for the retrieval encoder in this process: auto | cuda | cpu
# Override with env: PARAVERAG_TORCH_DEVICE=cuda
TORCH_DEVICE = os.environ.get("PARAVERAG_TORCH_DEVICE", "auto").strip().lower()

LLM_URL = os.environ.get("PARAVERAG_LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = "llama3.2-3b"
LLM_TEMPERATURE = 0.1
# QA often needs short lists or a few sentences; too low truncates before the model states supported facts.
LLM_MAX_TOKENS = 256
# Long wiki snippets + full gold answer in feedback can exceed local server context → HTTP 400.
LLM_MAX_USER_CHARS = 14_000
MAX_CHARS_PER_RETRIEVED_DOC = 2_000
MAX_GROUND_TRUTH_FEEDBACK_CHARS = 1_500

# Same HTTP LLM as generation; separate low temperature / token cap for accept/reject judging.
LLM_JUDGE_TEMPERATURE = 0.0
LLM_JUDGE_MAX_TOKENS = 128
# Thread pool size for parallel triple-judge stage (consistency / contradiction / hallucination).
# Override: PARAVERAG_EVAL_PARALLEL_JUDGE_WORKERS
EVAL_PARALLEL_JUDGE_WORKERS = 3
_workers_env = os.environ.get("PARAVERAG_EVAL_PARALLEL_JUDGE_WORKERS", "").strip()
if _workers_env:
    EVAL_PARALLEL_JUDGE_WORKERS = max(1, int(_workers_env))
EVAL_PARALLEL_JUDGE_WORKERS = min(EVAL_PARALLEL_JUDGE_WORKERS, CPU_MAX_THREADS)

# Verification scoring (see paraverrag/verification/pipeline.py and *_agent.py).
# strict — all three judges must return PASS for the verification step to succeed.
# lenient — at least VERIFICATION_MIN_PASS_VOTES of the three must PASS (default majority: 2 of 3).
# Override mode: PARAVERAG_VERIFICATION_PASS_MODE=strict|lenient
# Override votes: PARAVERAG_VERIFICATION_MIN_PASS_VOTES (1–3; ignored for strict except as upper bound)
VERIFICATION_PASS_MODE = os.environ.get("PARAVERAG_VERIFICATION_PASS_MODE", "lenient").strip().lower()
if VERIFICATION_PASS_MODE not in ("strict", "lenient"):
    VERIFICATION_PASS_MODE = "lenient"
_VERIFICATION_MIN_ENV = os.environ.get("PARAVERAG_VERIFICATION_MIN_PASS_VOTES", "").strip()
VERIFICATION_MIN_PASS_VOTES = 2
if _VERIFICATION_MIN_ENV:
    VERIFICATION_MIN_PASS_VOTES = max(1, min(3, int(_VERIFICATION_MIN_ENV)))

MAX_CORRECTION_ATTEMPTS = 3
# Sample counts for EVAL_SAMPLE_SOURCE hub/chroma only (not used when json + EVAL_JSON_MAX_ITEMS is None).
TEST_SET_SIZE = 50
RANDOM_SEED = 42
# Eval test set: JSON array of {"question": "...", "answer": "..."} (ground_truth accepted instead of answer).
EVAL_JSON_PATH = os.environ.get("PARAVERAG_EVAL_JSON", "./eval_test_set.json")
# When EVAL_SAMPLE_SOURCE=json: cap how many rows to run (None = use entire file; typical after populate).
# Env: PARAVERAG_EVAL_JSON_MAX_ITEMS or EVAL_JSON_MAX_ITEMS (either name).
EVAL_JSON_MAX_ITEMS: int | None = None
# Where to draw (question, gold answer) pairs for evaluation:
#   json   — load EVAL_JSON_PATH (default).
#   chroma — random rows from Chroma (needs question metadata on ingest).
#   hub    — reservoir sample from HF parquet streams (local cache when available).
EVAL_SAMPLE_SOURCE = os.environ.get("PARAVERAG_EVAL_SAMPLE_SOURCE", "json").strip().lower()
# Hub only (EVAL_SAMPLE_SOURCE=hub or evaluation.py --from-hub): stop scanning the HF parquet
# stream after this many valid QA rows when building the reservoir sample. Ignored for json/chroma.
# For json, cap eval rows with EVAL_JSON_MAX_ITEMS instead.
EVAL_SUBSET_MAX_VALID_ROWS: int | None = 50

# Bulk Chroma ingest: avoid default sync every 1k vectors (very slow on large corpora).
CHROMA_HNSW_SYNC_THRESHOLD = 600_000
CHROMA_HNSW_INTERNAL_BATCH = 20_000
# Lower = faster index build, slightly lower ANN quality (fine for MVP).
CHROMA_HNSW_CONSTRUCTION_EF = 64
# SentenceTransformer encode batch size on GPU (raise if you have headroom).
POPULATE_ENCODE_BATCH_SIZE = 1024
# Cap rows ingested from the stitched train dataset (None = full corpus). Smaller = faster MVP / less disk.
# Override with env: POPULATE_MAX_ROWS or PARAVERAG_POPULATE_MAX_ROWS (either name).
POPULATE_MAX_ROWS: int | None = None
# If True, store each query in Chroma metadata (useful for EVAL_SAMPLE_SOURCE=chroma).
# Override with env: POPULATE_STORE_QUESTION_METADATA or PARAVERAG_POPULATE_STORE_QUESTION_METADATA (true/false/1/0).
POPULATE_STORE_QUESTION_METADATA = False
# How many encoded batches may wait while the writer thread persists to Chroma (memory vs overlap).
POPULATE_PIPELINE_QUEUE_DEPTH = 2
# After ingest, write eval JSON: this percent of ingested rows (same slice as Chroma), drawn without replacement.
# Example: POPULATE_MAX_ROWS=10000 and PERCENTAGE_TEST_DATA=10 → 1000 eval rows in EVAL_JSON_PATH.
PERCENTAGE_TEST_DATA = 10.0
# If True, populate (paraverrag.populate) writes EVAL_JSON_PATH from the ingested slice (see PERCENTAGE_TEST_DATA).
POPULATE_WRITE_EVAL_JSON = True

_POPULATE_ROWS_ENV = (
    os.environ.get("PARAVERAG_POPULATE_MAX_ROWS", "").strip()
    or os.environ.get("POPULATE_MAX_ROWS", "").strip()
)
if _POPULATE_ROWS_ENV:
    POPULATE_MAX_ROWS = int(_POPULATE_ROWS_ENV)

_meta_env = (
    os.environ.get("PARAVERAG_POPULATE_STORE_QUESTION_METADATA", "").strip().lower()
    or os.environ.get("POPULATE_STORE_QUESTION_METADATA", "").strip().lower()
)
if _meta_env in ("1", "true", "yes", "on"):
    POPULATE_STORE_QUESTION_METADATA = True
elif _meta_env in ("0", "false", "no", "off"):
    POPULATE_STORE_QUESTION_METADATA = False

_pct_env = os.environ.get("PARAVERAG_PERCENTAGE_TEST_DATA", "").strip()
if _pct_env:
    PERCENTAGE_TEST_DATA = float(_pct_env)

_pop_eval_env = os.environ.get("PARAVERAG_POPULATE_WRITE_EVAL_JSON", "").strip().lower()
if _pop_eval_env in ("0", "false", "no", "off"):
    POPULATE_WRITE_EVAL_JSON = False
elif _pop_eval_env in ("1", "true", "yes", "on"):
    POPULATE_WRITE_EVAL_JSON = True

_eval_json_max_env = (
    os.environ.get("PARAVERAG_EVAL_JSON_MAX_ITEMS", "").strip()
    or os.environ.get("EVAL_JSON_MAX_ITEMS", "").strip()
)
if _eval_json_max_env:
    EVAL_JSON_MAX_ITEMS = int(_eval_json_max_env)

# Per-sample results CSV from paraverrag.evaluation (one row per sample × judge mode: query_id, dataset, question,
# gold_answer, predicted_answer, em, f1, hallucination_flagged, latencies, retries, *_fired, mode; optional trace_json).
# Set PARAVERAG_EVAL_RESULTS_CSV to empty to disable. Relative paths are from the process cwd.
_EVAL_CSV_ENV = os.environ.get("PARAVERAG_EVAL_RESULTS_CSV", "./eval_results.csv").strip()
EVAL_RESULTS_CSV_PATH: str | None = _EVAL_CSV_ENV if _EVAL_CSV_ENV else None
# If True, add a trace_json column with the full self-correction trace (can be large).
_EVAL_TRACE_ENV = os.environ.get("PARAVERAG_EVAL_CSV_INCLUDE_TRACE", "1").strip().lower()
EVAL_RESULTS_CSV_INCLUDE_TRACE = _EVAL_TRACE_ENV not in ("0", "false", "no", "off")
