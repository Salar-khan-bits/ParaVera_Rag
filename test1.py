"""
ParaVerRAG - Parallel Verification RAG Pipeline
Connects to your local LLaMA.cpp server at localhost:8080
Run: pip install sentence-transformers numpy requests --break-system-packages
Then: python paraverrag.py
"""

import time
import json
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
LLM_URL   = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "local-model"
TOP_K     = 3
MAX_RETRY = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Sample knowledge base (replace with your corpus) ─────────────────────────
DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) combines a retrieval mechanism with a language model. Documents are fetched from a corpus and passed as context to the LLM before generation.",
    "Hallucination in LLMs refers to the model generating factually incorrect or unsupported statements. Verification agents can check outputs against retrieved documents to reduce hallucinations.",
    "Cosine similarity measures the angle between two embedding vectors. It ranges from -1 to 1 and is widely used in semantic search to rank document relevance to a query.",
    "OpenCL is an open standard for parallel programming across CPUs, GPUs, and other processors. It enables heterogeneous computing without requiring NVIDIA-specific hardware.",
    "Natural Questions (NQ) is a benchmark dataset containing real Google search queries paired with Wikipedia answers, widely used to evaluate open-domain QA systems.",
]

# ── Embedding + Retrieval ────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

def embed(texts: list[str]) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Pure numpy cosine sim (swap this for your OpenCL kernel)."""
    return doc_vecs @ query_vec  # vecs already L2-normalized

def retrieve(query: str, docs: list[str], top_k: int = TOP_K):
    doc_vecs  = embed(docs)
    query_vec = embed([query])[0]

    # ── GPU scoring hook ──────────────────────────────────────────────────
    # Replace the line below with your OpenCL kernel call, e.g.:
    #   scores = opencl_cosine_sim(query_vec, doc_vecs)
    scores = cosine_similarity(query_vec, doc_vecs)
    # ─────────────────────────────────────────────────────────────────────

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(docs[i], float(scores[i])) for i in top_idx]

# ── LLM call ────────────────────────────────────────────────────────────────
def llm(prompt: str, max_tokens: int = 400) -> str:
    res = requests.post(LLM_URL, json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }, timeout=60)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

# ── Generation ───────────────────────────────────────────────────────────────
def generate(query: str, retrieved_docs: list[tuple], feedback: str = "") -> str:
    ctx = "\n".join(f"[{i+1}] {doc}" for i, (doc, _) in enumerate(retrieved_docs))
    correction = f"\n\nNote: A previous answer was flagged — {feedback}. Please fix this." if feedback else ""
    prompt = f"""You are a helpful assistant. Answer using only the provided context.

Context:
{ctx}

Question: {query}{correction}

Answer:"""
    return llm(prompt)

# ── Verification agents (run in parallel) ────────────────────────────────────
def _agent_call(name: str, query: str, answer: str, ctx: str) -> dict:
    prompts = {
        "consistency": f"""Does the answer stay consistent with the context throughout?
Context: {ctx}
Question: {query}
Answer: {answer}
Reply ONLY with JSON: {{"pass": true/false, "reason": "one sentence"}}""",

        "contradiction": f"""Does the answer contradict anything in the context?
Context: {ctx}
Question: {query}
Answer: {answer}
Reply ONLY with JSON: {{"pass": true/false, "reason": "one sentence"}}""",

        "hallucination": f"""Does the answer contain claims not supported by the context?
Context: {ctx}
Question: {query}
Answer: {answer}
Reply ONLY with JSON: {{"pass": true/false, "reason": "one sentence"}}""",
    }
    raw = llm(prompts[name], max_tokens=100)
    try:
        return {"agent": name, **json.loads(raw[raw.find("{"):raw.rfind("}")+1])}
    except Exception:
        return {"agent": name, "pass": True, "reason": "parse error — assuming pass"}

def verify(query: str, answer: str, retrieved_docs: list[tuple]) -> dict:
    ctx = " ".join(doc for doc, _ in retrieved_docs)
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_agent_call, name, query, answer, ctx): name
            for name in ["consistency", "contradiction", "hallucination"]
        }
        for future in as_completed(futures):
            r = future.result()
            results[r["agent"]] = r
            print(f"  [{r['agent'].upper()}] {'PASS' if r['pass'] else 'FAIL'} — {r['reason']}")
    return results

# ── Self-correction loop ─────────────────────────────────────────────────────
def self_correct(query: str, answer: str, verification: dict) -> str:
    failed = [r for r in verification.values() if not r["pass"]]
    if not failed:
        return ""
    return "; ".join(f"{r['agent']}: {r['reason']}" for r in failed)

# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(query: str, docs: list[str] = DOCUMENTS) -> str:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1 & 2: Parallel retrieval + GPU scoring
    print("\n[1/4] Retrieving and scoring documents...")
    retrieved = retrieve(query, docs)
    for doc, score in retrieved:
        print(f"  score={score:.3f} | {doc[:70]}...")

    # Step 3: Generate
    print("\n[2/4] Generating answer...")
    answer = generate(query, retrieved)
    print(f"  Draft: {answer[:120]}...")

    # Step 4: Verification + self-correction loop
    feedback = ""
    for attempt in range(1, MAX_RETRY + 1):
        print(f"\n[3/4] Verifying (attempt {attempt}/{MAX_RETRY})...")
        verification = verify(query, answer, retrieved)

        all_pass = all(r["pass"] for r in verification.values())
        if all_pass:
            print("  All agents passed.")
            break

        feedback = self_correct(query, answer, verification)
        if attempt < MAX_RETRY:
            print(f"\n[RETRY] Regenerating with feedback: {feedback}")
            answer = generate(query, retrieved, feedback=feedback)
        else:
            print("\n[MAX RETRIES] Returning best available answer.")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Final answer ({elapsed:.1f}s):\n{answer}")
    print(f"{'='*60}\n")
    return answer


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG and how does it reduce hallucinations?"
    run(query)