"""
Evaluation metrics for comparing Vanilla RAG vs GraphRAG.

MRR (Mean Reciprocal Rank):
    For each query, find the rank of the first relevant document.
    Score = 1/rank. Mean across all queries.

Top-K Accuracy:
    Fraction of queries where the correct answer appears in the top-K retrieved chunks.

Context Recall:
    Fraction of queries where retrieved context contains the ground-truth keywords.
"""

import re
from typing import Sequence


def reciprocal_rank(retrieved: list[str], ground_truth: str) -> float:
    """RR for a single query. Checks if ground truth keywords appear in retrieved passages."""
    gt_tokens = set(_tokenize(ground_truth))
    for rank, passage in enumerate(retrieved, start=1):
        passage_tokens = set(_tokenize(passage))
        overlap = gt_tokens & passage_tokens
        # Generous match: ≥30% of ground truth tokens present
        if len(gt_tokens) > 0 and len(overlap) / len(gt_tokens) >= 0.30:
            return 1.0 / rank
    return 0.0


def mrr(queries_results: list[tuple[list[str], str]]) -> float:
    """Mean Reciprocal Rank across all queries.

    Args:
        queries_results: list of (retrieved_passages, ground_truth_answer)
    Returns:
        MRR score in [0, 1]
    """
    if not queries_results:
        return 0.0
    return sum(reciprocal_rank(r, gt) for r, gt in queries_results) / len(queries_results)


def top_k_accuracy(queries_results: list[tuple[list[str], str]], k: int = 5) -> float:
    """Fraction of queries where ground truth is found in top-k retrieved passages."""
    if not queries_results:
        return 0.0
    hits = 0
    for retrieved, ground_truth in queries_results:
        gt_tokens = set(_tokenize(ground_truth))
        for passage in retrieved[:k]:
            passage_tokens = set(_tokenize(passage))
            overlap = gt_tokens & passage_tokens
            if len(gt_tokens) > 0 and len(overlap) / len(gt_tokens) >= 0.30:
                hits += 1
                break
    return hits / len(queries_results)


def context_recall(queries_results: list[tuple[list[str], str]]) -> float:
    """Fraction of queries where any retrieved passage contains the ground truth keywords."""
    return top_k_accuracy(queries_results, k=999)  # check all retrieved passages


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]  # ignore short stopwords
