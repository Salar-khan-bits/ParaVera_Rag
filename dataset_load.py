"""Utilities for loading Natural Questions data from Hugging Face."""

from __future__ import annotations

from typing import Dict, List

from datasets import load_dataset

HF_DATASET_NAME = "sentence-transformers/natural-questions"


def load_nq_examples(limit: int = 100, split: str = "train") -> List[Dict[str, str]]:
    dataset = load_dataset(HF_DATASET_NAME, split=split)
    rows: List[Dict[str, str]] = []
    for item in dataset.select(range(min(limit, len(dataset)))):
        rows.append(
            {
                "question": str(item["query"]).strip(),
                "answer": str(item["answer"]).strip(),
            }
        )
    return rows


def build_corpus_from_nq(examples: List[Dict[str, str]]) -> List[str]:
    # For this dataset format, answer strings act as evidence passages.
    return [ex["answer"] for ex in examples if ex["answer"]]


if __name__ == "__main__":
    sample = load_nq_examples(limit=5)
    print(f"Loaded {len(sample)} examples from {HF_DATASET_NAME}")
    for idx, row in enumerate(sample, start=1):
        print(f"{idx}. Q: {row['question']}")
        print(f"   A: {row['answer'][:100]}...")