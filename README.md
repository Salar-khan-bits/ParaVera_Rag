# ParaVerRAG (MVP)

RAG pipeline over **Chroma** with **sentence-transformers** retrieval, an **OpenAI-compatible HTTP LLM** for answers, and optional **triple LLM judges** (consistency, contradiction, hallucination) with self-correction.

## Layout

```
paraverrag/
  __init__.py       # package version
  config.py         # settings (env overrides)
  cli.py            # question / verify CLI
  rag.py            # retrieve, prompt, generate, self-correction
  llm_client.py     # HTTP LLM + judge calls
  data.py           # HF hub + JSON + Chroma eval loaders
  evaluation.py     # benchmarks + CSV metrics
  populate.py       # bulk Chroma ingest (+ optional NPZ shards)
  retrieval/        # sharded parallel CPU search; optional OpenCL dot kernel
  verification/     # judge agents + pipeline (serial / parallel)
```

Legacy scripts at the repo root (`main.py`, `evaluation.py`, `populate_chroma.py`) delegate to this package so existing commands keep working.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Start an OpenAI-compatible server (llama.cpp, Ollama, vLLM, etc.) and point `PARAVERAG_LLM_URL` at it if not using the default `http://localhost:8080/v1/chat/completions`.

## Commands

After `pip install -e .`:

| Command | Purpose |
|--------|---------|
| `paraverrag "your question"` | RAG answer |
| `paraverrag-eval` | Run evaluation (see `--help`) |
| `paraverrag-ingest` | Populate Chroma from the HF corpus |

Equivalent modules:

```bash
python -m paraverrag.cli "your question"
python -m paraverrag.evaluation --help
python -m paraverrag.populate
```

## Configuration

See `paraverrag/config.py`. Common environment variables:

- `PARAVERAG_LLM_URL`, `PARAVERAG_TORCH_DEVICE`, `PARAVERAG_CPU_MAX_THREADS`, `PARAVERAG_OPENCL_DEVICE` (`cpu` \| `gpu` \| `auto`), `PARAVERAG_EVAL_RESULTS_CSV`, `PARAVERAG_VERIFICATION_PASS_MODE`, etc.

### Retrieval backends

| Mode | Env | Behavior |
|------|-----|----------|
| **Chroma HNSW** (default) | `PARAVERAG_RETRIEVAL_BACKEND=chroma` | One embedding + one Chroma query per question. |
| **Sharded** | `PARAVERAG_RETRIEVAL_BACKEND=sharded` | Uses `./retrieval_shards` by default (`RETRIEVAL_SHARD_DIR`): documents are split across **non-overlapping** NPZ shards at ingest. **Cosine similarity** vs shard rows runs in **OpenCL** (`opencl_cosine.py`, one shared context), shard-by-shard. Set `PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1` only if you must fall back to **parallel CPU NumPy** matmul (still capped by `PARAVERAG_CPU_MAX_THREADS`). |

**OpenCL (default for sharded cosine):** `pip install pyopencl` (or `pip install -e ".[opencl]"`) plus an **ICD** so `clinfo` shows at least one platform. If `Number of platforms` is **0**, PyOpenCL cannot run kernels: install `nvidia-opencl-icd` (NVIDIA), `pocl-opencl-icd` (CPU), or your vendor’s OpenCL package. Until then, set **`PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1`** or use **`PARAVERAG_RETRIEVAL_BACKEND=chroma`**.

**Device selection:** By default **`PARAVERAG_OPENCL_DEVICE=cpu`** — the first OpenCL **CPU** device (typically [POCL](http://portablecl.org/)) is used so sharded cosine does not run on the GPU OpenCL stack. Install `pocl-opencl-icd` for that path. If you only have a GPU ICD and want it for cosine, set **`PARAVERAG_OPENCL_DEVICE=auto`** (first device the loader picks) or **`gpu`** (first GPU device only).

**Thread cap:** `PARAVERAG_CPU_MAX_THREADS` limits logical threads for **NumPy shard workers** and **parallel eval judges** (defaults to `os.cpu_count()`). `PARAVERAG_RETRIEVAL_CPU_WORKERS` further caps shard worker count (≤ shard count and ≤ `CPU_MAX_THREADS`).

Ingest writes shard files when `POPULATE_WRITE_RETRIEVAL_SHARDS` is on (default `true`; disable with `PARAVERAG_POPULATE_WRITE_RETRIEVAL_SHARDS=0`). Tune shard count with `PARAVERAG_RETRIEVAL_SHARD_COUNT`.

## Data

- **Ingest** writes vectors under `./chroma_db` (configurable) and, by default, **embedding shards** under `./retrieval_shards` for the sharded backend.
- **Eval** can use `eval_test_set.json`, Chroma samples, or Hugging Face parquet streams (`EVAL_SAMPLE_SOURCE`).

## What this MVP delivers (vs easy-to-overstate claims)

Use this table for demos or write-ups so expectations match the code.

| Topic | Reality in this repo |
|--------|----------------------|
| **End-to-end speedup (serial vs parallel judges)** | Parallel **verification** (three judge LLM calls) saves a lot of wall time vs serial judges; **overall** end-to-end speedup is often **~1.4–1.6×** when retrieval is already cheap (Chroma HNSW + small `TOP_K`). A **2–3×** overall claim only holds if retrieval/generation dominated; here retrieval is usually not the bottleneck. |
| **“Hallucination rate below 7%”** | The **hallucination judge** is an LLM rubric over retrieved text + gold; its FAIL rate is **not** the same as a calibrated production hallucination metric. On tiny evals (e.g. a handful of questions), **one** failure can read as **20%** — too few samples to support a population claim. Report **PASS/Fail counts and sample size**, not a false-precision percentage. |
| **Parallel CPU retrieval / shard workers** | **Implemented** when `RETRIEVAL_BACKEND=sharded` + `PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1`: round-robin shard **NPZ** at ingest; **parallel CPU** (`ThreadPoolExecutor`, ≤`PARAVERAG_CPU_MAX_THREADS`) scores shards with NumPy. Default sharded path uses **OpenCL** for cosine, not CPU matmul. |
| **OpenCL cosine / dot kernel** | **Implemented (default for sharded):** cosine = OpenCL dot on L2-normalized rows (`opencl_cosine.py`). NumPy only with `PARAVERAG_RETRIEVAL_ALLOW_NUMPY_COSINE=1`. CSV **`scoring_latency_s`** remains **LLM generation** time, not retrieval. |
| **Parallel verification agents** | **Implemented:** `parallel` mode runs consistency, contradiction, and hallucination judge prompts **concurrently** (thread pool + HTTP). Eval reports **judge wall vs Σ per-call** savings — that is the strongest parallel result. |
| **Self-correction (up to 3 attempts)** | **Implemented** (`MAX_CORRECTION_ATTEMPTS`). Often **attempts = 1** if judges pass on the first answer; the loop is real even when retries are rare. |
| **Consumer / CPU-friendly** | The pipeline can run **CPU-only** for the encoder (configurable) and uses a **local HTTP LLM**; no custom datacenter stack. Optional **CUDA** speeds up `sentence-transformers` if available. |

## License

MIT
