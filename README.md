# ParaVerRAG

Parallel, verifiable RAG pipeline with six staged execution blocks:

1. Query encoding
2. Parallel shard retrieval (multi-worker)
3. GPU-style scoring (OpenCL-aware fallback)
4. LLM answer generation
5. Parallel verification agents
6. Self-correction loop (max retries) and verified output

## Run

```bash
python main.py "What is the capital of France?"
```

## Evaluate

```bash
python evaluation/run_eval.py
```
