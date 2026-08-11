"""External Medical MIRAGE benchmark adapter and exact-choice evaluator.

MIRAGE requires question-only retrieval (QOR): answer options may be shown to
the generator, but must not be sent to the retriever. This module represents
those inputs separately and checkpoints every paid model call.

The official benchmark and generated outputs are stored below
``graphrag/eval/external/``, which is intentionally gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


MIRAGE_URL = (
    "https://raw.githubusercontent.com/gzxiong/MIRAGE/main/benchmark.json"
)
MIRAGE_SHA256 = "6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa"
ANSWER_PARSER_VERSION = 3
DATASET_ORDER = ("mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq")
OFFICIAL_COUNTS = {
    "mmlu": 1089,
    "medqa": 1273,
    "medmcqa": 4183,
    "pubmedqa": 500,
    "bioasq": 618,
}
DEFAULT_CACHE = Path(__file__).parent / "external" / "mirage" / "benchmark.json"
DEFAULT_OUTPUT = Path(__file__).parent / "external" / "mirage" / "results.json"
DEFAULT_TEXTBOOKS_OUTPUT = (
    Path(__file__).parent / "external" / "mirage" / "textbooks_results.json"
)
DEFAULT_TEXTBOOKS_AGENT_OUTPUT = (
    Path(__file__).parent / "external" / "mirage" / "textbooks_agent_results.json"
)
MODEL_PRICING_USD_PER_MILLION = {
    "gemini-2.5-flash": {
        "input": 0.30,
        "output_including_thinking": 2.50,
        "as_of": "2026-08-11",
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "tier": "paid",
    }
}

TEXTBOOKS_GENERATION_INSTRUCTION = """You are a helpful medical expert answering
a multiple-choice medical question using the relevant documents. Think through
the evidence and choose a definite answer from the provided options. Return JSON
with keys `step_by_step_thinking` and `answer_choice`."""


@dataclass(frozen=True)
class MirageCase:
    case_id: str
    dataset: str
    source_id: str
    retrieval_query: str
    answer_query: str
    options: dict[str, str]
    answer: str


@dataclass
class SystemResult:
    case_id: str
    dataset: str
    source_id: str
    system: str
    gold_choice: str
    prediction: str
    correct: bool
    raw_answer: str
    latency_s: float
    status: str
    error: str
    tool_names: list[str]
    tool_errors: list[str]
    retry_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_latency_s: float = 0.0
    context_count: int = 0
    validation_verdicts: list[str] = field(default_factory=list)
    reflection_approved: bool = False


SystemRunner = Callable[[MirageCase], Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_benchmark(
    destination: Path = DEFAULT_CACHE,
    url: str = MIRAGE_URL,
    expected_sha256: str = MIRAGE_SHA256,
) -> Path:
    """Download the pinned official file and reject silent upstream changes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _sha256(destination)
        if actual != expected_sha256:
            raise ValueError(
                f"Existing MIRAGE file has SHA-256 {actual}, expected {expected_sha256}. "
                "Move it aside before downloading the pinned version."
            )
        return destination

    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(url, headers={"User-Agent": "medical-graphrag-eval/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded MIRAGE file has SHA-256 {actual}, expected {expected_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _answer_query(question: str, options: Mapping[str, str]) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in options.items())
    labels = ", ".join(options)
    return (
        f"Medical multiple-choice question:\n{question}\n\nOptions:\n{choices}\n\n"
        "Use the retrieved evidence and choose the single best option. End the "
        f"answer with `FINAL_ANSWER: <choice>`, where <choice> is one of {labels}."
    )


def load_benchmark(path: Path, enforce_official_counts: bool = True) -> list[MirageCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("MIRAGE root must be a JSON object")

    cases: list[MirageCase] = []
    for dataset in DATASET_ORDER:
        rows = raw.get(dataset)
        if not isinstance(rows, dict):
            raise ValueError(f"MIRAGE dataset {dataset!r} must be an object")
        if enforce_official_counts and len(rows) != OFFICIAL_COUNTS[dataset]:
            raise ValueError(
                f"Unexpected {dataset} count: {len(rows)} != {OFFICIAL_COUNTS[dataset]}"
            )
        for source_id, row in rows.items():
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row {dataset}:{source_id}")
            question = str(row.get("question") or "").strip()
            options = row.get("options")
            answer = str(row.get("answer") or "").strip().upper()
            if not question or not isinstance(options, dict) or len(options) < 2:
                raise ValueError(f"Missing question/options in {dataset}:{source_id}")
            normalized_options = {
                str(label).strip().upper(): str(text).strip()
                for label, text in options.items()
            }
            if any(not label or not text for label, text in normalized_options.items()):
                raise ValueError(f"Empty option in {dataset}:{source_id}")
            if answer not in normalized_options:
                raise ValueError(f"Invalid answer in {dataset}:{source_id}: {answer!r}")
            cases.append(
                MirageCase(
                    case_id=f"{dataset}:{source_id}",
                    dataset=dataset,
                    source_id=str(source_id),
                    retrieval_query=question,
                    answer_query=_answer_query(question, normalized_options),
                    options=normalized_options,
                    answer=answer,
                )
            )
    return cases


def _stable_seed(seed: int, dataset: str) -> int:
    value = hashlib.sha256(f"{seed}:{dataset}".encode()).hexdigest()[:16]
    return int(value, 16)


def stratified_sample(
    cases: list[MirageCase], limit: int = 500, seed: int = 13
) -> list[MirageCase]:
    """Allocate cases as evenly as possible across the five MIRAGE datasets."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(cases):
        raise ValueError(f"limit {limit} exceeds available cases {len(cases)}")

    grouped = {
        dataset: [case for case in cases if case.dataset == dataset]
        for dataset in DATASET_ORDER
    }
    quotas = {dataset: limit // len(DATASET_ORDER) for dataset in DATASET_ORDER}
    for dataset in DATASET_ORDER[: limit % len(DATASET_ORDER)]:
        quotas[dataset] += 1
    if any(quotas[name] > len(grouped[name]) for name in DATASET_ORDER):
        raise ValueError("A requested stratum is larger than its source dataset")

    selected: list[MirageCase] = []
    for dataset in DATASET_ORDER:
        rng = random.Random(_stable_seed(seed, dataset))
        selected.extend(rng.sample(grouped[dataset], quotas[dataset]))
    return selected


def dataset_fingerprint(cases: list[MirageCase]) -> str:
    serialized = "\n".join(
        json.dumps(
            [case.case_id, case.retrieval_query, case.options, case.answer],
            sort_keys=True,
            ensure_ascii=False,
        )
        for case in cases
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def parse_answer_choice(text: str, choices: set[str]) -> str:
    """Extract an explicit choice without guessing from arbitrary prose."""
    normalized_choices = {choice.upper() for choice in choices}
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for key in ("answer_choice", "prediction", "answer"):
            value = str(payload.get(key, "")).strip().upper()
            if value in normalized_choices:
                return value

    boxed = re.findall(r"(?i)\\?boxed\s*\{\s*([A-Z])\s*\}", text)
    valid_boxed = [
        value.upper() for value in boxed if value.upper() in normalized_choices
    ]
    if valid_boxed:
        return valid_boxed[-1]

    explicit = re.findall(
        r"(?i)(?:final[_ ]answer|answer[_ ]choice|answer|choice)"
        r"\s*(?:is\s*)?(?::|=)?\s*[`*\"']*\(?([A-Z])\)?",
        text,
    )
    valid = [value.upper() for value in explicit if value.upper() in normalized_choices]
    if valid:
        return valid[-1]

    compact = re.sub(r"[`*_\s]", "", text).upper()
    match = re.fullmatch(r"\(?([A-Z])\)?[.)]?", compact)
    if match and match.group(1) in normalized_choices:
        return match.group(1)
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _usage(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def build_closed_book_runner(model: str | None = None) -> SystemRunner:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from graphrag.config import (
        GEMINI_MAX_RETRIES,
        GEMINI_MODEL,
        GEMINI_REQUEST_TIMEOUT_S,
        GOOGLE_API_KEY,
    )

    llm = ChatGoogleGenerativeAI(
        model=model or GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
    )

    def run(case: MirageCase) -> Mapping[str, Any]:
        prompt = (
            "Answer this medical multiple-choice benchmark question. Do not use "
            "external tools. Return one JSON object only in the form "
            '{"answer_choice":"A"}.\n\n'
            + case.answer_query
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": _content_text(response.content), **_usage(response)}

    return run


def build_agent_runner() -> SystemRunner:
    from langchain_core.messages import ToolMessage
    from graphrag.agent.react_agent import AgentState, build_agent

    graph = build_agent()

    def run(case: MirageCase) -> Mapping[str, Any]:
        trace_id = uuid.uuid4()
        state = graph.invoke(
            AgentState(
                query=case.answer_query,
                retrieval_query=case.retrieval_query,
                messages=[],
                retrieved_context=[],
                retry_count=0,
                final_answer="",
                candidate_answers=[],
                validation_verdicts=[],
                status="running",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                retrieved_ids=[],
                retrieval_latency_s=0.0,
                context_count=0,
            ),
            config={
                "recursion_limit": 20,
                "run_id": trace_id,
                "run_name": "medical-mirage-case",
                "tags": ["stage-5c", "mirage", case.dataset],
                "metadata": {
                    "evaluation_type": "external_medical_mirage",
                    "case_id": case.case_id,
                    "question_only_retrieval": True,
                },
            },
        )
        messages = state.get("messages", [])
        tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
        verdicts = list(state.get("validation_verdicts", []))
        return {
            "answer": state.get("final_answer", ""),
            "status": state.get("status", "unknown"),
            "tool_names": [item.name for item in tool_messages if item.name],
            "tool_errors": [
                _content_text(item.content)
                for item in tool_messages
                if "Tool unavailable" in _content_text(item.content)
            ],
            "retry_count": int(state.get("retry_count", 0)),
            "validation_verdicts": verdicts,
            "reflection_approved": any(
                verdict.startswith("APPROVED") for verdict in verdicts
            ),
            "retrieved_ids": list(state.get("retrieved_ids", [])),
            "retrieval_latency_s": float(state.get("retrieval_latency_s", 0.0)),
            "context_count": int(state.get("context_count", 0)),
            "trace_id": str(trace_id),
            "input_tokens": int(state.get("input_tokens", 0)),
            "output_tokens": int(state.get("output_tokens", 0)),
            "total_tokens": int(state.get("total_tokens", 0)),
        }

    return run


def build_textbooks_rag_runner(
    index_path: Path, top_k: int = 8, model: str | None = None
) -> SystemRunner:
    """Build a matched-corpus RAG baseline using official MedRAG Textbooks."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from graphrag.config import (
        GEMINI_MAX_RETRIES,
        GEMINI_MODEL,
        GEMINI_REQUEST_TIMEOUT_S,
        GOOGLE_API_KEY,
    )
    from graphrag.eval.mirage_corpus import TextbooksBM25Retriever

    retriever = TextbooksBM25Retriever(index_path)
    llm = ChatGoogleGenerativeAI(
        model=model or GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
    )

    def run(case: MirageCase) -> Mapping[str, Any]:
        return _run_textbooks_rag_case(case, retriever, llm, HumanMessage, top_k)

    return run


def _format_textbook_context(snippets: list[Any]) -> str:
    return "\n\n".join(
        f"Document [{position}] (Title: {snippet.title}) {snippet.content}"
        for position, snippet in enumerate(snippets, start=1)
    )


def _build_textbooks_retrieval_tool(retriever: Any, top_k: int) -> Any:
    """Create an artifact-producing LangChain tool over the matched corpus."""
    from langchain_core.tools import tool

    @tool("retrieve_textbooks_bm25", response_format="content_and_artifact")
    def retrieve_textbooks_bm25(query: str) -> tuple[str, dict[str, Any]]:
        """Retrieve top medical textbook passages for a question-only query."""
        started = time.perf_counter()
        snippets = retriever.retrieve(query, k=top_k)
        latency = time.perf_counter() - started
        content = _format_textbook_context(snippets)
        if not content:
            content = "No relevant textbook passages found."
        return content, {
            "retrieved_ids": [snippet.snippet_id for snippet in snippets],
            "retrieval_latency_s": latency,
            "context_count": len(snippets),
        }

    return retrieve_textbooks_bm25


def build_textbooks_agent_runner(
    index_path: Path, top_k: int = 8, model: str | None = None
) -> SystemRunner:
    """Build a LangGraph Agent over the same Textbooks BM25 configuration."""
    from langchain_core.messages import ToolMessage
    from graphrag.agent.react_agent import AgentState, RetrievalStep, build_agent
    from graphrag.config import GEMINI_MODEL
    from graphrag.eval.mirage_corpus import TextbooksBM25Retriever

    retriever = TextbooksBM25Retriever(index_path)
    retrieval_tool = _build_textbooks_retrieval_tool(retriever, top_k)
    retrieval_step = RetrievalStep(
        tool_name="retrieve_textbooks_bm25",
        build_args=lambda query: {"query": query},
        instruction="Retrieve the textbook evidence exactly once before answering.",
    )
    graph = build_agent(
        tools=[retrieval_tool],
        retrieval_steps=(retrieval_step,),
        system_prompt=TEXTBOOKS_GENERATION_INSTRUCTION,
        enable_safety=False,
        allow_optional_tool_calls=False,
        max_tool_calls=1,
        model=model or GEMINI_MODEL,
    )

    def run(case: MirageCase) -> Mapping[str, Any]:
        trace_id = uuid.uuid4()
        state = graph.invoke(
            AgentState(
                query=case.answer_query,
                retrieval_query=case.retrieval_query,
                messages=[],
                retrieved_context=[],
                retry_count=0,
                final_answer="",
                candidate_answers=[],
                validation_verdicts=[],
                status="running",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                retrieved_ids=[],
                retrieval_latency_s=0.0,
                context_count=0,
            ),
            config={
                "recursion_limit": 20,
                "run_id": trace_id,
                "run_name": "medical-mirage-textbooks-agent-case",
                "tags": ["stage-5g", "mirage", "textbooks-agent", case.dataset],
                "metadata": {
                    "evaluation_type": "external_medical_mirage",
                    "case_id": case.case_id,
                    "question_only_retrieval": True,
                    "generation_model": model or GEMINI_MODEL,
                },
            },
        )
        messages = state.get("messages", [])
        tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
        verdicts = list(state.get("validation_verdicts", []))
        return {
            "answer": state.get("final_answer", ""),
            "status": state.get("status", "unknown"),
            "tool_names": [item.name for item in tool_messages if item.name],
            "tool_errors": [
                _content_text(item.content)
                for item in tool_messages
                if "Tool unavailable" in _content_text(item.content)
            ],
            "retry_count": int(state.get("retry_count", 0)),
            "validation_verdicts": verdicts,
            "reflection_approved": any(
                verdict.startswith("APPROVED") for verdict in verdicts
            ),
            "retrieved_ids": list(state.get("retrieved_ids", [])),
            "retrieval_latency_s": float(state.get("retrieval_latency_s", 0.0)),
            "context_count": int(state.get("context_count", 0)),
            "input_tokens": int(state.get("input_tokens", 0)),
            "output_tokens": int(state.get("output_tokens", 0)),
            "total_tokens": int(state.get("total_tokens", 0)),
            "trace_id": str(trace_id),
        }

    return run


def _run_textbooks_rag_case(
    case: MirageCase,
    retriever: Any,
    llm: Any,
    human_message_class: Any,
    top_k: int,
) -> Mapping[str, Any]:
    """Pure orchestration seam used to verify question-only retrieval offline."""
    retrieval_started = time.perf_counter()
    snippets = retriever.retrieve(case.retrieval_query, k=top_k)
    retrieval_latency = time.perf_counter() - retrieval_started
    context = _format_textbook_context(snippets)
    prompt = (
        TEXTBOOKS_GENERATION_INSTRUCTION
        + "\n\n"
        f"Relevant documents:\n{context}\n\n{case.answer_query}"
    )
    response = llm.invoke([human_message_class(content=prompt)])
    return {
        "answer": _content_text(response.content),
        "status": "completed",
        "tool_names": ["retrieve_textbooks_bm25"],
        "retrieved_ids": [snippet.snippet_id for snippet in snippets],
        "retrieval_latency_s": retrieval_latency,
        "context_count": len(snippets),
        **_usage(response),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def summarize_system(
    results: list[SystemResult],
    pricing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(results)
    correct = sum(result.correct for result in results)
    total_input_tokens = sum(result.input_tokens for result in results)
    total_output_tokens = sum(result.output_tokens for result in results)
    per_dataset: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        subset = [result for result in results if result.dataset == dataset]
        if not subset:
            continue
        hits = sum(result.correct for result in subset)
        per_dataset[dataset] = {
            "n": len(subset),
            "accuracy": hits / len(subset),
            "invalid_choice_rate": sum(not result.prediction for result in subset)
            / len(subset),
            "error_rate": sum(bool(result.error) for result in subset) / len(subset),
        }
    accuracies = [value["accuracy"] for value in per_dataset.values()]
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_95ci": _wilson_interval(correct, total),
        "macro_dataset_accuracy": statistics.fmean(accuracies) if accuracies else 0.0,
        "invalid_choice_rate": sum(not result.prediction for result in results) / total
        if total
        else 0.0,
        "error_rate": sum(bool(result.error) for result in results) / total
        if total
        else 0.0,
        "p50_latency_s": _percentile([result.latency_s for result in results], 0.50),
        "p95_latency_s": _percentile([result.latency_s for result in results], 0.95),
        "average_total_tokens": statistics.fmean(
            result.total_tokens for result in results
        )
        if results
        else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": sum(result.total_tokens for result in results),
        "estimated_cost_usd": (
            total_input_tokens * float(pricing["input"])
            + total_output_tokens * float(pricing["output_including_thinking"])
        )
        / 1_000_000
        if pricing
        else None,
        "tool_sequence_success_rate": statistics.fmean(
            float(
                "retrieve_graph" in result.tool_names
                and "retrieve_vector" in result.tool_names
            )
            for result in results
        )
        if results
        and any(
            "retrieve_graph" in result.tool_names
            or "retrieve_vector" in result.tool_names
            for result in results
        )
        else None,
        "retrieval_success_rate": statistics.fmean(
            float(result.context_count > 0) for result in results
        )
        if results and any("retrieve_textbooks_bm25" in result.tool_names for result in results)
        else None,
        "average_retrieval_latency_s": statistics.fmean(
            result.retrieval_latency_s for result in results
        )
        if results and any("retrieve_textbooks_bm25" in result.tool_names for result in results)
        else None,
        "average_context_count": statistics.fmean(
            result.context_count for result in results
        )
        if results and any("retrieve_textbooks_bm25" in result.tool_names for result in results)
        else None,
        "reflection_approval_rate": statistics.fmean(
            float(result.reflection_approved) for result in results
        )
        if results and any(result.validation_verdicts for result in results)
        else None,
        "average_retry_count": statistics.fmean(
            result.retry_count for result in results
        )
        if results and any(result.validation_verdicts for result in results)
        else None,
        "per_dataset": per_dataset,
    }


def _mcnemar_exact_p(agent_wins: int, baseline_wins: int) -> float:
    discordant = agent_wins + baseline_wins
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(agent_wins, baseline_wins) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_comparison(
    baseline: list[SystemResult], candidate: list[SystemResult]
) -> dict[str, Any]:
    baseline_by_id = {result.case_id: result for result in baseline}
    candidate_by_id = {result.case_id: result for result in candidate}
    paired_ids = sorted(baseline_by_id.keys() & candidate_by_id.keys())
    agent_wins = sum(
        candidate_by_id[key].correct and not baseline_by_id[key].correct
        for key in paired_ids
    )
    baseline_wins = sum(
        baseline_by_id[key].correct and not candidate_by_id[key].correct
        for key in paired_ids
    )
    candidate_accuracy = (
        sum(candidate_by_id[key].correct for key in paired_ids) / len(paired_ids)
        if paired_ids
        else 0.0
    )
    baseline_accuracy = (
        sum(baseline_by_id[key].correct for key in paired_ids) / len(paired_ids)
        if paired_ids
        else 0.0
    )
    return {
        "n_pairs": len(paired_ids),
        "candidate_wins": agent_wins,
        "baseline_wins": baseline_wins,
        "ties": len(paired_ids) - agent_wins - baseline_wins,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "mcnemar_exact_p": _mcnemar_exact_p(agent_wins, baseline_wins),
    }


def _report(
    cases: list[MirageCase],
    results: list[SystemResult],
    systems: list[str],
    seed: int,
    generation_model: str,
    source_path: Path,
    system_metadata: Mapping[str, Any] | None = None,
    pricing: Mapping[str, Any] | None = None,
    reuse_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    by_system = {
        system: [result for result in results if result.system == system]
        for system in systems
    }
    comparisons = {}
    if "closed-book" in by_system:
        for system in systems:
            if system != "closed-book":
                comparisons[f"{system}_vs_closed-book"] = paired_comparison(
                    by_system["closed-book"], by_system[system]
                )
    return {
        "evaluation_type": "external_medical_mirage_exact_choice",
        "benchmark": {
            "name": "MIRAGE: Medical Information Retrieval-Augmented Generation Evaluation",
            "source_url": MIRAGE_URL,
            "source_sha256": _sha256(source_path),
            "question_only_retrieval": True,
            "n_selected": len(cases),
            "selection_seed": seed,
            "selection_fingerprint": dataset_fingerprint(cases),
            "counts": {
                dataset: sum(case.dataset == dataset for case in cases)
                for dataset in DATASET_ORDER
            },
        },
        "generation_model": generation_model,
        "pricing_usd_per_million_tokens": dict(pricing) if pricing else None,
        "answer_parser_version": ANSWER_PARSER_VERSION,
        "systems": systems,
        "system_metadata": dict(system_metadata or {}),
        "corpus_note": (
            "current-agent uses this project's small guideline FAISS/Neo4j corpus. "
            "textbooks-rag uses the official MedRAG Textbooks corpus with a local "
            "SQLite FTS5 BM25 retriever. Different corpora/retrievers must be "
            "reported separately from published MedRAG configurations."
        ),
        "metrics": {
            system: summarize_system(by_system[system], pricing)
            for system in systems
        },
        "paired_comparisons": comparisons,
        "reused_checkpoint": dict(reuse_metadata) if reuse_metadata else None,
        "results": [asdict(result) for result in results],
    }


def _load_reusable_results(
    path: Path,
    cases: list[MirageCase],
    systems: list[str],
    generation_model: str,
) -> dict[tuple[str, str], SystemResult]:
    """Import compatible results from another report without repeating API calls."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("benchmark", {}).get("selection_fingerprint") != dataset_fingerprint(
        cases
    ):
        raise ValueError("Reusable checkpoint uses a different case selection")
    if saved.get("generation_model") != generation_model:
        raise ValueError("Reusable checkpoint uses a different generation model")

    case_by_id = {case.case_id: case for case in cases}
    parser_changed = saved.get("answer_parser_version") != ANSWER_PARSER_VERSION
    reusable: dict[tuple[str, str], SystemResult] = {}
    for item in saved.get("results", []):
        result = SystemResult(**item)
        case = case_by_id.get(result.case_id)
        if case is None or result.system not in systems:
            continue
        if (
            result.dataset != case.dataset
            or result.source_id != case.source_id
            or result.gold_choice != case.answer
        ):
            raise ValueError(f"Reusable result does not match case {result.case_id}")
        if parser_changed and not result.error:
            result.prediction = parse_answer_choice(
                result.raw_answer, set(case.options)
            )
            result.correct = result.prediction == result.gold_choice
        reusable[(result.case_id, result.system)] = result
    return reusable


def run_benchmark(
    cases: list[MirageCase],
    runners: Mapping[str, SystemRunner],
    output_path: Path,
    source_path: Path,
    seed: int = 13,
    generation_model: str = "configured-gemini-model",
    resume: bool = True,
    system_metadata: Mapping[str, Any] | None = None,
    reuse_results_from: Path | None = None,
) -> dict[str, Any]:
    systems = list(runners)
    pricing = MODEL_PRICING_USD_PER_MILLION.get(generation_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[tuple[str, str], SystemResult] = {}
    reuse_metadata = None
    if reuse_results_from is not None:
        if not reuse_results_from.exists():
            raise FileNotFoundError(
                f"Reusable checkpoint not found: {reuse_results_from}"
            )
        imported = _load_reusable_results(
            reuse_results_from, cases, systems, generation_model
        )
        previous.update(imported)
        reuse_metadata = {
            "source": str(reuse_results_from),
            "matched_results": len(imported),
            "saved_model_calls": len(imported),
        }
    if resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        benchmark = saved.get("benchmark", {})
        if benchmark.get("selection_fingerprint") != dataset_fingerprint(cases):
            raise ValueError("Existing checkpoint uses a different case selection")
        if saved.get("systems") != systems:
            raise ValueError("Existing checkpoint uses different systems")
        if saved.get("generation_model") != generation_model:
            raise ValueError("Existing checkpoint uses a different generation model")
        if saved.get("system_metadata", {}) != dict(system_metadata or {}):
            raise ValueError("Existing checkpoint uses different system metadata")
        case_by_id = {case.case_id: case for case in cases}
        parser_changed = saved.get("answer_parser_version") != ANSWER_PARSER_VERSION
        for item in saved.get("results", []):
            result = SystemResult(**item)
            # Preserve completed paid calls, but allow transient provider errors
            # to be retried on resume instead of becoming permanent outcomes.
            if result.error:
                continue
            if parser_changed and result.case_id in case_by_id and not result.error:
                case = case_by_id[result.case_id]
                result.prediction = parse_answer_choice(
                    result.raw_answer, set(case.options)
                )
                result.correct = result.prediction == result.gold_choice
            previous[(result.case_id, result.system)] = result

    results: list[SystemResult] = []
    total_calls = len(cases) * len(systems)
    position = 0
    for case in cases:
        for system, runner in runners.items():
            position += 1
            key = (case.case_id, system)
            if key in previous:
                results.append(previous[key])
                continue
            started = time.perf_counter()
            try:
                output = dict(runner(case))
                raw_answer = _content_text(output.get("answer", ""))
                prediction = parse_answer_choice(raw_answer, set(case.options))
                error = ""
            except Exception as exc:
                output = {}
                raw_answer = ""
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
            result = SystemResult(
                case_id=case.case_id,
                dataset=case.dataset,
                source_id=case.source_id,
                system=system,
                gold_choice=case.answer,
                prediction=prediction,
                correct=prediction == case.answer,
                raw_answer=raw_answer,
                latency_s=time.perf_counter() - started,
                status=str(output.get("status", "error" if error else "completed")),
                error=error,
                tool_names=list(output.get("tool_names", [])),
                tool_errors=list(output.get("tool_errors", [])),
                retry_count=int(output.get("retry_count", 0)),
                input_tokens=int(output.get("input_tokens", 0)),
                output_tokens=int(output.get("output_tokens", 0)),
                total_tokens=int(output.get("total_tokens", 0)),
                retrieved_ids=list(output.get("retrieved_ids", [])),
                retrieval_latency_s=float(output.get("retrieval_latency_s", 0.0)),
                context_count=int(output.get("context_count", 0)),
                validation_verdicts=list(output.get("validation_verdicts", [])),
                reflection_approved=bool(output.get("reflection_approved", False)),
            )
            results.append(result)
            report = _report(
                cases,
                results,
                systems,
                seed,
                generation_model,
                source_path,
                system_metadata,
                pricing,
                reuse_metadata,
            )
            output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            outcome = "ERROR" if error else "correct" if result.correct else "wrong"
            print(f"[{position}/{total_calls}] {case.case_id} {system}: {outcome}")
    final = _report(
        cases,
        results,
        systems,
        seed,
        generation_model,
        source_path,
        system_metadata,
        pricing,
        reuse_metadata,
    )
    output_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def _dry_run(cases: list[MirageCase], path: Path, seed: int) -> dict[str, Any]:
    return {
        "benchmark": "Medical MIRAGE",
        "source_sha256": _sha256(path),
        "official_total": sum(OFFICIAL_COUNTS.values()),
        "n_selected": len(cases),
        "selection_seed": seed,
        "selection_fingerprint": dataset_fingerprint(cases),
        "counts": {
            dataset: sum(case.dataset == dataset for case in cases)
            for dataset in DATASET_ORDER
        },
        # QOR is guaranteed structurally: runners receive retrieval_query and
        # answer_query as separate fields. Option text may naturally occur in a
        # question (for example, the word "no"), so substring checks are invalid.
        "question_only_retrieval": True,
        "retrieval_input_field": "retrieval_query",
        "generation_input_field": "answer_query",
        "external_calls": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--reuse-results-from",
        type=Path,
        help="Import compatible system/case results from a prior report.",
    )
    from graphrag.eval.mirage_corpus import DEFAULT_INDEX as DEFAULT_TEXTBOOKS_INDEX

    parser.add_argument("--textbooks-index", type=Path, default=DEFAULT_TEXTBOOKS_INDEX)
    parser.add_argument("--textbooks-top-k", type=int, default=8)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=(
            "closed-book",
            "current-agent",
            "textbooks-rag",
            "textbooks-agent",
        ),
        default=("closed-book", "current-agent"),
    )
    args = parser.parse_args(argv)

    if args.download:
        download_benchmark(args.dataset)
    if not args.dataset.exists():
        parser.error("MIRAGE data not found; rerun with --download")
    if _sha256(args.dataset) != MIRAGE_SHA256:
        parser.error("MIRAGE source hash does not match the pinned official file")

    cases = stratified_sample(load_benchmark(args.dataset), args.limit, args.seed)
    if args.dry_run:
        print(json.dumps(_dry_run(cases, args.dataset, args.seed), indent=2))
        return

    from graphrag.config import GEMINI_MODEL
    from graphrag.eval.mirage_corpus import index_metadata

    runners: dict[str, SystemRunner] = {}
    system_metadata: dict[str, Any] = {}
    for system in args.systems:
        if system == "closed-book":
            runners[system] = build_closed_book_runner(GEMINI_MODEL)
        elif system == "current-agent":
            runners[system] = build_agent_runner()
            system_metadata[system] = {
                "corpus": "project-guidelines-v1",
                "retrieval": "mandatory graph then vector",
            }
        elif system == "textbooks-rag":
            runners[system] = build_textbooks_rag_runner(
                args.textbooks_index, args.textbooks_top_k, GEMINI_MODEL
            )
            system_metadata[system] = {
                "corpus": "MedRAG/textbooks",
                "retriever": "sqlite-fts5-bm25",
                "top_k": args.textbooks_top_k,
                "index": index_metadata(args.textbooks_index),
            }
        elif system == "textbooks-agent":
            runners[system] = build_textbooks_agent_runner(
                args.textbooks_index, args.textbooks_top_k, GEMINI_MODEL
            )
            system_metadata[system] = {
                "corpus": "MedRAG/textbooks",
                "retriever": "sqlite-fts5-bm25",
                "top_k": args.textbooks_top_k,
                "index": index_metadata(args.textbooks_index),
                "orchestrator": "LangGraph reason/act/validate",
                "max_reflection_retries": 3,
            }
    if "textbooks-agent" in args.systems and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_TEXTBOOKS_AGENT_OUTPUT
    elif "textbooks-rag" in args.systems and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_TEXTBOOKS_OUTPUT
    report = run_benchmark(
        cases,
        runners,
        args.output,
        args.dataset,
        seed=args.seed,
        generation_model=GEMINI_MODEL,
        resume=not args.no_resume,
        system_metadata=system_metadata,
        reuse_results_from=args.reuse_results_from,
    )
    print("\nMedical MIRAGE")
    print("=" * 48)
    for system, metrics in report["metrics"].items():
        print(
            f"{system:<16} accuracy={metrics['accuracy']:.3f} "
            f"invalid={metrics['invalid_choice_rate']:.3f} "
            f"p95={metrics['p95_latency_s']:.2f}s"
        )
    print(f"\nLocal checkpoint: {args.output}")


if __name__ == "__main__":
    main()
