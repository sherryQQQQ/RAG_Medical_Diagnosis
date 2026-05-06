"""
Benchmark: Vanilla RAG vs GraphRAG
-----------------------------------
Runs both pipelines on the same test cases from medical_generalization.csv
and prints a side-by-side comparison table.

Usage:
    python -m graphrag.eval.benchmark
"""

import csv
import json
import time
from dataclasses import dataclass, field

from graphrag.config import EVAL_CSV_PATH
from graphrag.eval.metrics import mrr, top_k_accuracy, context_recall
from graphrag.retrieval.hybrid import HybridRetriever


# ---------- Vanilla RAG baseline (reuse existing final/ code) -----------------

def _load_vanilla_pipeline():
    """Import the existing txtai-based retriever from final/."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "final"))
    from prepare import DocumentVectorizer
    from retrieve import DocumentRetriever
    dv = DocumentVectorizer()
    embeddings = dv.load_database()
    dr = DocumentRetriever(embeddings)
    return dr


def run_vanilla(retriever, query: str) -> tuple[list[str], float]:
    start = time.time()
    results = retriever.retrieve(query, limit=5)
    elapsed = time.time() - start
    passages = [text for _, text in results]
    return passages, elapsed


# ---------- GraphRAG pipeline -------------------------------------------------

def run_graphrag(hybrid: HybridRetriever, query: str, symptoms: list[str]) -> tuple[list[str], float]:
    start = time.time()
    results = hybrid.retrieve(query=query, symptoms=symptoms)
    elapsed = time.time() - start
    passages = [r.text for r in results]
    return passages, elapsed


# ---------- Load test data ---------------------------------------------------

@dataclass
class TestCase:
    prompt: str
    answer: str
    symptoms: list[str] = field(default_factory=list)


def load_test_cases(path: str, rows: tuple[int, int] = (70, 80)) -> list[TestCase]:
    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if rows[0] <= i < rows[1]:
                # Heuristic: extract symptom-like words from the prompt
                import re
                prompt = row.get("prompt", "")
                answer = row.get("answer", "")
                # Simple symptom extraction: look for noun phrases after "with" or "has"
                symptoms = re.findall(
                    r"(?:with|has|presents?|complains? of|reports?)\s+([a-z ,]+?)(?:\.|,|and|$)",
                    prompt.lower(),
                )
                flat_symptoms = [s.strip() for part in symptoms for s in part.split(",") if s.strip()]
                cases.append(TestCase(prompt=prompt, answer=answer, symptoms=flat_symptoms[:5]))
    return cases


# ---------- Main benchmark ---------------------------------------------------

def run_benchmark():
    print("Loading test cases...")
    test_cases = load_test_cases(EVAL_CSV_PATH)
    if not test_cases:
        print("[ERROR] No test cases loaded — check EVAL_CSV_PATH and row range.")
        return

    print(f"Running benchmark on {len(test_cases)} cases...\n")

    # --- Vanilla ---
    try:
        vanilla_retriever = _load_vanilla_pipeline()
        vanilla_results = []
        vanilla_latencies = []
        for tc in test_cases:
            passages, elapsed = run_vanilla(vanilla_retriever, tc.prompt)
            vanilla_results.append((passages, tc.answer))
            vanilla_latencies.append(elapsed)
        vanilla_mrr = mrr(vanilla_results)
        vanilla_top5 = top_k_accuracy(vanilla_results, k=5)
        vanilla_recall = context_recall(vanilla_results)
        vanilla_avg_latency = sum(vanilla_latencies) / len(vanilla_latencies)
    except Exception as e:
        print(f"[WARN] Vanilla pipeline failed: {e}")
        vanilla_mrr = vanilla_top5 = vanilla_recall = vanilla_avg_latency = float("nan")

    # --- GraphRAG ---
    try:
        hybrid = HybridRetriever()
        graph_results = []
        graph_latencies = []
        for tc in test_cases:
            passages, elapsed = run_graphrag(hybrid, tc.prompt, tc.symptoms)
            graph_results.append((passages, tc.answer))
            graph_latencies.append(elapsed)
        graph_mrr = mrr(graph_results)
        graph_top5 = top_k_accuracy(graph_results, k=5)
        graph_recall = context_recall(graph_results)
        graph_avg_latency = sum(graph_latencies) / len(graph_latencies)
    except Exception as e:
        print(f"[WARN] GraphRAG pipeline failed: {e}")
        graph_mrr = graph_top5 = graph_recall = graph_avg_latency = float("nan")

    # --- Print results ---
    _print_table(
        vanilla=(vanilla_mrr, vanilla_top5, vanilla_recall, vanilla_avg_latency),
        graphrag=(graph_mrr, graph_top5, graph_recall, graph_avg_latency),
        n=len(test_cases),
    )

    # --- Save JSON ---
    results_path = EVAL_CSV_PATH.replace("medical_generalization.csv", "eval_results.json")
    output = {
        "n_cases": len(test_cases),
        "vanilla_rag": {"mrr": vanilla_mrr, "top5_acc": vanilla_top5, "context_recall": vanilla_recall, "avg_latency_s": vanilla_avg_latency},
        "graphrag": {"mrr": graph_mrr, "top5_acc": graph_top5, "context_recall": graph_recall, "avg_latency_s": graph_avg_latency},
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def _print_table(vanilla, graphrag, n):
    cols = ["MRR", "Top-5 Acc", "Context Recall", "Avg Latency(s)"]
    v = [f"{x:.3f}" if x == x else "n/a" for x in vanilla]
    g = [f"{x:.3f}" if x == x else "n/a" for x in graphrag]

    print(f"\n{'='*65}")
    print(f"  BENCHMARK RESULTS  (n={n} test cases)")
    print(f"{'='*65}")
    print(f"{'Metric':<20} {'Vanilla RAG':>15} {'GraphRAG+Agent':>15}")
    print(f"{'-'*65}")
    for col, vi, gi in zip(cols, v, g):
        print(f"{col:<20} {vi:>15} {gi:>15}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_benchmark()
