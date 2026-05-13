"""
Synthetic retrieval benchmark derived from the original project dataset.

This module avoids Gemini, Neo4j, FAISS, and downloaded embedding models. It
first builds a deterministic synthetic retrieval dataset from
final/medical_generalization.csv, then compares:

1. Vector-only lexical retrieval: a lightweight stand-in for vanilla RAG.
2. Hybrid graph+vector retrieval: vector ranking fused with graph-term matches
   using Reciprocal Rank Fusion.

Run:
    python -m graphrag.main generate-synthetic-data
    python -m graphrag.main synthetic-benchmark
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Iterable


RRF_K = 60
GRAPH_WEIGHT = 2.0

EVAL_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(EVAL_DIR))
ORIGINAL_DATASET_PATH = os.path.join(REPO_ROOT, "final", "medical_generalization.csv")
DEFAULT_SYNTHETIC_DATA_PATH = os.path.join(
    EVAL_DIR,
    "data",
    "synthetic_medical_retrieval.json",
)


STOPWORDS = {
    "a", "about", "according", "after", "all", "also", "an", "and", "any",
    "are", "as", "at", "based", "be", "been", "before", "but", "by",
    "can", "care", "case", "clinical", "clinicians", "current", "days",
    "do", "does", "for", "from", "guideline", "guidelines", "has", "have",
    "he", "her", "his", "how", "in", "is", "it", "man", "mg", "most",
    "new", "next", "no", "of", "old", "on", "or", "patient", "patients",
    "plan", "prescribe", "recommendation", "recommended", "reports", "result",
    "scheduled", "should", "she", "test", "testing", "the", "therapy",
    "this", "to", "treatment", "what", "when", "which", "who", "with",
    "woman", "year", "years", "you",
}


@dataclass(frozen=True)
class ClinicalCase:
    disease: str
    label: str
    symptoms: tuple[str, ...]
    aliases: tuple[str, ...]
    treatment: str
    document: str
    source_prompt: str
    source_answer: str
    original_id: str

    @property
    def graph_terms(self) -> tuple[str, ...]:
        return self.symptoms + self.aliases + (self.label,)


@dataclass(frozen=True)
class SyntheticQuery:
    query: str
    expected_disease: str
    query_type: str


@dataclass(frozen=True)
class RetrievalResult:
    disease: str
    score: float
    source: str
    label: str = ""


@dataclass(frozen=True)
class MethodMetrics:
    mrr: float
    top1: float
    top3: float


@dataclass(frozen=True)
class ComparisonReport:
    n_queries: int
    vector_only: MethodMetrics
    hybrid_graph_vector: MethodMetrics
    sample_queries: list[dict]


def load_original_dataset(path: str = ORIGINAL_DATASET_PATH) -> list[dict]:
    """Load the original medical_generalization.csv rows."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            row for row in reader
            if row.get("prompt", "").strip() and row.get("answer", "").strip()
        ]


def build_synthetic_cases(source_path: str = ORIGINAL_DATASET_PATH) -> list[ClinicalCase]:
    """Turn each original QA row into a retrieval case."""
    rows = load_original_dataset(source_path)
    cases = []
    for i, row in enumerate(rows):
        original_id = row.get("id") or row.get("", str(i)) or str(i)
        case_id = f"case_{int(original_id):03d}" if str(original_id).isdigit() else f"case_{i:03d}"
        prompt = row["prompt"].strip()
        answer = row["answer"].strip()
        answer_before = row.get("answer_before", "").strip()

        prompt_terms = extract_key_terms(prompt, max_terms=12)
        answer_terms = extract_key_terms(answer, max_terms=12)
        label_terms = short_terms(answer_terms)[:2] or short_terms(prompt_terms)[:2] or (case_id,)
        label = f"{case_id}: {' / '.join(label_terms)}"

        document = f"Updated guideline answer: {answer}"
        if answer_before:
            document += f"\nPrevious baseline answer: {answer_before}"

        cases.append(
            ClinicalCase(
                disease=case_id,
                label=label,
                symptoms=prompt_terms,
                aliases=answer_terms,
                treatment=answer,
                document=document,
                source_prompt=prompt,
                source_answer=answer,
                original_id=str(original_id),
            )
        )
    return cases


def generate_synthetic_queries(
    cases: list[ClinicalCase] | None = None,
    n_per_case: int = 3,
    seed: int = 13,
) -> list[SyntheticQuery]:
    """Generate deterministic query variants from original prompts and answers."""
    cases = cases or build_synthetic_cases()
    rng = random.Random(seed)
    queries: list[SyntheticQuery] = []

    for case in cases:
        prompt_terms = list(case.symptoms)
        answer_terms = list(case.aliases)
        variants = [
            ("original_prompt", case.source_prompt),
            (
                "prompt_terms",
                "Which updated guideline applies when the case includes "
                f"{format_terms(rng, prompt_terms)}?",
            ),
            (
                "answer_terms",
                "Find the recommendation involving "
                f"{format_terms(rng, answer_terms)}.",
            ),
            (
                "delta_from_baseline",
                "Which original dataset item changed toward "
                f"{format_terms(rng, answer_terms[:6] + prompt_terms[:6])}?",
            ),
        ]
        for query_type, query in variants[:n_per_case]:
            queries.append(
                SyntheticQuery(
                    query=query,
                    expected_disease=case.disease,
                    query_type=query_type,
                )
            )

    rng.shuffle(queries)
    return queries


def build_synthetic_dataset(
    source_path: str = ORIGINAL_DATASET_PATH,
    n_per_case: int = 3,
    seed: int = 13,
) -> dict:
    """Construct a complete synthetic retrieval dataset from the original CSV."""
    cases = build_synthetic_cases(source_path)
    queries = generate_synthetic_queries(cases, n_per_case=n_per_case, seed=seed)
    return {
        "metadata": {
            "name": "synthetic_medical_retrieval",
            "description": (
                "Deterministic synthetic retrieval dataset derived from "
                "final/medical_generalization.csv."
            ),
            "source_dataset": source_path,
            "seed": seed,
            "n_per_case": n_per_case,
            "n_cases": len(cases),
            "n_queries": len(queries),
        },
        "cases": [asdict(case) for case in cases],
        "queries": [asdict(query) for query in queries],
    }


def save_synthetic_dataset(
    path: str = DEFAULT_SYNTHETIC_DATA_PATH,
    source_path: str = ORIGINAL_DATASET_PATH,
    n_per_case: int = 3,
    seed: int = 13,
) -> str:
    """Generate synthetic data from the original CSV and persist it as JSON."""
    dataset = build_synthetic_dataset(source_path=source_path, n_per_case=n_per_case, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")
    return path


def load_synthetic_dataset(path: str = DEFAULT_SYNTHETIC_DATA_PATH) -> tuple[list[ClinicalCase], list[SyntheticQuery]]:
    """Load synthetic cases and queries from the generated JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    cases = [
        ClinicalCase(
            disease=row["disease"],
            label=row["label"],
            symptoms=tuple(row["symptoms"]),
            aliases=tuple(row["aliases"]),
            treatment=row["treatment"],
            document=row["document"],
            source_prompt=row["source_prompt"],
            source_answer=row["source_answer"],
            original_id=row["original_id"],
        )
        for row in dataset["cases"]
    ]
    queries = [
        SyntheticQuery(
            query=row["query"],
            expected_disease=row["expected_disease"],
            query_type=row["query_type"],
        )
        for row in dataset["queries"]
    ]
    return cases, queries


def extract_key_terms(text: str, max_terms: int = 12) -> tuple[str, ...]:
    """Extract deterministic terms from original prompts/answers."""
    candidates: list[str] = []

    for match in re.findall(r"\b[A-Z][A-Z0-9/-]{1,}\b", text):
        candidates.append(match.lower())

    for match in re.findall(r"\b\d+(?:[./-]\d+)*(?:\s?(?:mg|mmhg|mm hg|days?|hours?|years?|%|cells/ul|ng/ml))?", text.lower()):
        candidates.append(match.strip())

    tokens = token_list(text)
    candidates.extend(tokens)
    candidates.extend(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    candidates.extend(" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2))

    unique = []
    seen = set()
    for term in candidates:
        normalized = normalize_space(term)
        if len(normalized) < 3 or normalized in seen or normalized in STOPWORDS:
            continue
        seen.add(normalized)
        unique.append(normalized)

    unique.sort(key=lambda term: (-term_score(term), text.lower().find(term.split()[0]), term))
    return tuple(unique[:max_terms])


def format_terms(rng: random.Random, terms: list[str], n: int = 3) -> str:
    useful_terms = short_terms(terms) or [term for term in terms if term]
    if not useful_terms:
        return "the original clinical scenario"
    sample_size = min(n, len(useful_terms))
    selected = rng.sample(useful_terms, sample_size) if len(useful_terms) > sample_size else useful_terms
    return ", ".join(selected)


def short_terms(terms: Iterable[str]) -> list[str]:
    return [
        term for term in terms
        if term and (len(term.split()) <= 2 or any(ch.isdigit() for ch in term))
    ]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .,:;!?")


def token_list(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def tokenize(text: str) -> set[str]:
    return set(token_list(text))


def term_score(term: str) -> int:
    token_count = len(term.split())
    has_digit = int(any(ch.isdigit() for ch in term))
    return token_count * 10 + min(len(term), 20) + has_digit * 5


class VectorOnlyRetriever:
    """Tiny lexical retriever that represents the vanilla RAG baseline."""

    def __init__(self, cases: list[ClinicalCase]):
        self.cases = cases
        self.doc_tokens = {case.disease: tokenize(case.document) for case in cases}

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        results = []
        for case in self.cases:
            doc_tokens = self.doc_tokens[case.disease]
            overlap = query_tokens & doc_tokens
            denom = math.sqrt(max(len(query_tokens), 1) * max(len(doc_tokens), 1))
            score = len(overlap) / denom
            results.append(RetrievalResult(case.disease, score, "vector", case.label))

        return sorted(results, key=lambda r: (-r.score, r.disease))[:k]


class HybridGraphVectorRetriever:
    """Graph-term matches fused with vector ranks using RRF."""

    def __init__(self, cases: list[ClinicalCase]):
        self.cases = cases
        self.vector = VectorOnlyRetriever(cases)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        vector_ranked = self.vector.retrieve(query, k=len(self.cases))
        graph_ranked = self._graph_rank(query)
        fused: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(vector_ranked, start=1):
            score = 1.0 / (RRF_K + rank)
            fused[result.disease] = RetrievalResult(
                disease=result.disease,
                score=fused.get(result.disease, result).score + score if result.disease in fused else score,
                source="hybrid",
                label=result.label,
            )

        for rank, result in enumerate(graph_ranked, start=1):
            score = GRAPH_WEIGHT / (RRF_K + rank)
            existing = fused.get(result.disease)
            fused[result.disease] = RetrievalResult(
                disease=result.disease,
                score=(existing.score if existing else 0.0) + score,
                source="hybrid",
                label=result.label,
            )

        return sorted(fused.values(), key=lambda r: (-r.score, r.disease))[:k]

    def _graph_rank(self, query: str) -> list[RetrievalResult]:
        query_text = normalize_space(query)
        query_tokens = tokenize(query)
        results = []

        for case in self.cases:
            score = 0.0
            for term in case.graph_terms:
                normalized = normalize_space(term)
                if not normalized:
                    continue
                if normalized in query_text:
                    score += 2.0 + len(normalized.split()) * 0.25
                else:
                    term_tokens = set(normalized.split())
                    if term_tokens and term_tokens.issubset(query_tokens):
                        score += 1.0

            if score:
                results.append(RetrievalResult(case.disease, score, "graph", case.label))

        return sorted(results, key=lambda r: (-r.score, r.disease))


def reciprocal_rank(results: Iterable[RetrievalResult], expected_disease: str) -> float:
    for rank, result in enumerate(results, start=1):
        if result.disease == expected_disease:
            return 1.0 / rank
    return 0.0


def top_k_hit(results: Iterable[RetrievalResult], expected_disease: str, k: int) -> float:
    return float(any(r.disease == expected_disease for r in list(results)[:k]))


def evaluate(retriever, queries: list[SyntheticQuery], k: int = 5) -> MethodMetrics:
    ranked = [(retriever.retrieve(q.query, k=k), q.expected_disease) for q in queries]
    n = len(ranked)
    return MethodMetrics(
        mrr=sum(reciprocal_rank(results, expected) for results, expected in ranked) / n,
        top1=sum(top_k_hit(results, expected, 1) for results, expected in ranked) / n,
        top3=sum(top_k_hit(results, expected, 3) for results, expected in ranked) / n,
    )


def compare_methods(
    data_path: str = DEFAULT_SYNTHETIC_DATA_PATH,
    n_per_case: int = 3,
    seed: int = 13,
    create_if_missing: bool = True,
) -> ComparisonReport:
    if create_if_missing and not os.path.exists(data_path):
        save_synthetic_dataset(data_path, n_per_case=n_per_case, seed=seed)

    cases, queries = load_synthetic_dataset(data_path)
    vector = VectorOnlyRetriever(cases)
    hybrid = HybridGraphVectorRetriever(cases)
    labels = {case.disease: case.label for case in cases}

    samples = []
    for query in queries[:5]:
        vector_top = vector.retrieve(query.query, k=1)[0]
        hybrid_top = hybrid.retrieve(query.query, k=1)[0]
        samples.append(
            {
                "query": query.query,
                "expected": query.expected_disease,
                "expected_label": labels[query.expected_disease],
                "vector_top1": vector_top.disease,
                "vector_label": vector_top.label,
                "hybrid_top1": hybrid_top.disease,
                "hybrid_label": hybrid_top.label,
            }
        )

    return ComparisonReport(
        n_queries=len(queries),
        vector_only=evaluate(vector, queries),
        hybrid_graph_vector=evaluate(hybrid, queries),
        sample_queries=samples,
    )


def print_report(report: ComparisonReport) -> None:
    print("\nSynthetic Retrieval Comparison")
    print("=" * 58)
    print(f"Queries: {report.n_queries}")
    print(f"{'Method':<24} {'MRR':>8} {'Top-1':>8} {'Top-3':>8}")
    print("-" * 58)
    print(
        f"{'Vector-only baseline':<24} "
        f"{report.vector_only.mrr:>8.3f} "
        f"{report.vector_only.top1:>8.3f} "
        f"{report.vector_only.top3:>8.3f}"
    )
    print(
        f"{'Hybrid graph+vector':<24} "
        f"{report.hybrid_graph_vector.mrr:>8.3f} "
        f"{report.hybrid_graph_vector.top1:>8.3f} "
        f"{report.hybrid_graph_vector.top3:>8.3f}"
    )
    print("\nSample cases")
    print(json.dumps(report.sample_queries, indent=2))


def main() -> None:
    data_path = save_synthetic_dataset()
    print(f"Synthetic dataset written to: {data_path}")
    print_report(compare_methods(data_path=data_path))


if __name__ == "__main__":
    main()
