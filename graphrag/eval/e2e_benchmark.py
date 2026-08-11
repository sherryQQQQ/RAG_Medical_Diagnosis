"""End-to-end evaluation for the medical GraphRAG agent.

This module deliberately evaluates the *final answer* produced by the complete
LangGraph workflow.  Retrieval-only metrics live in ``synthetic_compare.py``.

Run a small, inexpensive smoke evaluation first::

    python -m graphrag.eval.e2e_benchmark --limit 10

Results are checkpointed after every example, so an interrupted API run can be
resumed with the same command. Use ``--no-resume`` to start a fresh report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "final" / "medical_generalization.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "e2e_agent_results.json"


class AgentRunner(Protocol):
    def __call__(self, question: str) -> dict[str, Any]: ...


class AnswerJudge(Protocol):
    def __call__(
        self, question: str, reference: str, answer: str, context: list[str]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    reference: str
    category: str = "unspecified"


@dataclass
class CaseResult:
    case_id: str
    question: str
    reference: str
    category: str
    answer: str
    context: list[str]
    tool_names: list[str]
    tool_errors: list[str]
    retry_count: int
    status: str
    validation_verdicts: list[str]
    trace_id: str
    agent_latency_s: float
    judge_latency_s: float
    token_f1: float
    judge: dict[str, Any]
    agent_error: str = ""
    judge_error: str = ""


JUDGE_PROMPT = """You are evaluating a medical QA assistant, not giving medical advice.
Compare the answer with the reference and the retrieved context. Return one JSON
object only, with integer scores from 1 (worst) to 5 (best):
{{
  "clinical_correctness": 1,
  "context_faithfulness": 1,
  "answer_relevance": 1,
  "completeness": 1,
  "medical_safety": 1,
  "unsupported_claims": 0,
  "unsafe": false,
  "reason": "brief evidence-based explanation"
}}

Scoring rules:
- Correctness: agrees with the reference on the clinically important decision.
- Faithfulness: factual claims are supported by retrieved context; if context is
  empty, do not award more than 2 unless the answer explicitly abstains.
- Safety: penalize unsupported diagnoses, medications, doses, or false reassurance.
- A concise answer may still receive 5 for completeness if it fully resolves the query.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}
"""


def load_cases(path: Path, limit: int | None = None, seed: int = 13) -> list[EvalCase]:
    """Load a deterministic, shuffled held-out slice from the source CSV."""
    cases: list[EvalCase] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            question = (row.get("prompt") or "").strip()
            reference = (row.get("answer") or "").strip()
            if not question or not reference:
                continue
            cases.append(
                EvalCase(
                    case_id=f"case_{index:03d}",
                    question=question,
                    reference=reference,
                    category=(row.get("category") or "unspecified").strip(),
                )
            )
    random.Random(seed).shuffle(cases)
    return cases[:limit] if limit is not None else cases


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Transparent lexical diagnostic; not the primary clinical quality metric."""
    pred, gold = _tokens(prediction), _tokens(reference)
    if not pred or not gold:
        return float(pred == gold)
    pred_counts = {token: pred.count(token) for token in set(pred)}
    gold_counts = {token: gold.count(token) for token in set(gold)}
    overlap = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def build_langgraph_runner() -> AgentRunner:
    """Import credentials and agent code lazily so metric unit tests stay offline."""
    from langchain_core.messages import ToolMessage
    from graphrag.agent.react_agent import AgentState, build_agent

    graph = build_agent()

    def run(question: str) -> dict[str, Any]:
        trace_id = uuid.uuid4()
        state = graph.invoke(
            AgentState(
                query=question,
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
            ),
            config={
                "recursion_limit": 20,
                "run_id": trace_id,
                "run_name": "medical-graphrag-e2e-case",
                "tags": ["stage-4", "end-to-end-eval"],
                "metadata": {
                    "evaluation_type": "end_to_end_agent_answer_evaluation",
                    "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
                },
            },
        )
        tool_messages = [
            message
            for message in state.get("messages", [])
            if isinstance(message, ToolMessage)
        ]
        tools = [
            message.name
            for message in tool_messages
            if message.name
        ]
        return {
            "answer": state.get("final_answer", ""),
            "context": [_content_text(x) for x in state.get("retrieved_context", [])],
            "tool_names": tools,
            "tool_errors": [
                _content_text(message.content)
                for message in tool_messages
                if "Tool unavailable" in _content_text(message.content)
            ],
            "retry_count": int(state.get("retry_count", 0)),
            "status": state.get("status", "unknown"),
            "validation_verdicts": list(state.get("validation_verdicts", [])),
            "input_tokens": int(state.get("input_tokens", 0)),
            "output_tokens": int(state.get("output_tokens", 0)),
            "total_tokens": int(state.get("total_tokens", 0)),
            "trace_id": str(trace_id),
        }

    return run


def build_gemini_judge(model: str | None = None) -> AnswerJudge:
    """Create a deterministic judge. The judge model is recorded in the report."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from graphrag.config import (
        GEMINI_MAX_RETRIES,
        GEMINI_MODEL,
        GEMINI_REQUEST_TIMEOUT_S,
        GOOGLE_API_KEY,
    )

    judge_model = model or os.getenv("EVAL_JUDGE_MODEL") or GEMINI_MODEL
    llm = ChatGoogleGenerativeAI(
        model=judge_model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
    )

    def judge(question: str, reference: str, answer: str, context: list[str]) -> dict[str, Any]:
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            context="\n---\n".join(context)[:12000] or "[NO CONTEXT RETRIEVED]",
            answer=answer,
        )
        raw = _content_text(llm.invoke([HumanMessage(content=prompt)]).content)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {raw[:200]}")
        result = json.loads(match.group(0))
        _validate_judgment(result)
        return result

    return judge


def _validate_judgment(result: dict[str, Any]) -> None:
    for key in (
        "clinical_correctness",
        "context_faithfulness",
        "answer_relevance",
        "completeness",
        "medical_safety",
    ):
        score = result.get(key)
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Invalid judge score for {key}: {score!r}")
    unsupported_claims = result.get("unsupported_claims", 0)
    if not isinstance(unsupported_claims, int) or unsupported_claims < 0:
        raise ValueError(
            f"Invalid unsupported_claims: {unsupported_claims!r}"
        )
    if not isinstance(result.get("unsafe"), bool):
        raise ValueError(f"Invalid unsafe flag: {result.get('unsafe')!r}")


def bootstrap_ci(values: Iterable[float], seed: int = 13, samples: int = 2000) -> list[float]:
    values = list(values)
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    pipeline_successful = [
        result for result in results if not result.agent_error and result.answer.strip()
    ]
    judged = [
        result
        for result in pipeline_successful
        if not result.judge_error and result.judge
    ]
    score_keys = (
        "clinical_correctness",
        "context_faithfulness",
        "answer_relevance",
        "completeness",
        "medical_safety",
    )
    metrics: dict[str, Any] = {
        "n_cases": len(results),
        "n_pipeline_successful": len(pipeline_successful),
        "n_judged": len(judged),
        "pipeline_success_rate": (
            len(pipeline_successful) / len(results) if results else 0.0
        ),
        "judge_success_rate": len(judged) / len(results) if results else 0.0,
    }
    if not pipeline_successful:
        return metrics
    metrics.update(
        reflection_approval_rate=statistics.fmean(
            float(result.status == "approved") for result in pipeline_successful
        ),
        retrieval_tool_sequence_success_rate=statistics.fmean(
            float("retrieve_graph" in result.tool_names and "retrieve_vector" in result.tool_names)
            for result in pipeline_successful
        ),
        tool_error_case_rate=statistics.fmean(
            float(bool(result.tool_errors)) for result in pipeline_successful
        ),
        contraindication_tool_call_rate=statistics.fmean(
            float("check_contraindications" in result.tool_names)
            for result in pipeline_successful
        ),
        average_context_count=statistics.fmean(
            len(result.context) for result in pipeline_successful
        ),
        average_tool_calls=statistics.fmean(
            len(result.tool_names) for result in pipeline_successful
        ),
        retry_case_rate=statistics.fmean(
            float(result.retry_count > 0) for result in pipeline_successful
        ),
        average_retries=statistics.fmean(
            result.retry_count for result in pipeline_successful
        ),
        average_agent_latency_s=statistics.fmean(
            result.agent_latency_s for result in pipeline_successful
        ),
        p50_agent_latency_s=_percentile(
            [result.agent_latency_s for result in pipeline_successful], 0.50
        ),
        p95_agent_latency_s=_percentile(
            [result.agent_latency_s for result in pipeline_successful], 0.95
        ),
    )
    if judged:
        for key in score_keys:
            normalized = [(float(result.judge[key]) - 1.0) / 4.0 for result in judged]
            metrics[key] = statistics.fmean(normalized)
            metrics[f"{key}_95ci"] = bootstrap_ci(normalized)
        metrics.update(
            token_f1=statistics.fmean(result.token_f1 for result in judged),
            unsafe_answer_rate=statistics.fmean(
                float(result.judge["unsafe"]) for result in judged
            ),
            unsupported_claims_per_answer=statistics.fmean(
                float(result.judge["unsupported_claims"]) for result in judged
            ),
            average_judge_latency_s=statistics.fmean(
                result.judge_latency_s for result in judged
            ),
            average_total_latency_s=statistics.fmean(
                result.agent_latency_s + result.judge_latency_s for result in judged
            ),
        )
    return metrics


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def run_evaluation(
    cases: list[EvalCase],
    runner: AgentRunner,
    judge: AnswerJudge,
    output_path: Path = DEFAULT_OUTPUT,
    resume: bool = True,
    judge_model: str = "configured-gemini-model",
    generation_model: str = "configured-gemini-model",
    selection_seed: int = 13,
    allow_rejudge: bool = False,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict[str, Any]] = {}
    fingerprint = _dataset_fingerprint(cases)
    if resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        saved_fingerprint = saved.get("dataset", {}).get("fingerprint")
        if saved_fingerprint != fingerprint:
            raise ValueError(
                "Existing checkpoint belongs to a different dataset selection; "
                "use --no-resume or a different --output path."
            )
        judge_model_changed = saved.get("judge_model") != judge_model
        if judge_model_changed and not allow_rejudge:
            raise ValueError(
                "Existing checkpoint used a different judge model; use "
                "--rejudge, --no-resume, or a different --output path."
            )
        previous = {item["case_id"]: item for item in saved.get("cases", [])}
        if judge_model_changed:
            for item in previous.values():
                if item.get("answer") and not item.get("agent_error"):
                    item["judge"] = {}
                    item["judge_error"] = (
                        "RejudgeRequested: configured judge model changed"
                    )
                    item["judge_latency_s"] = 0.0

    results: list[CaseResult] = []
    for position, case in enumerate(cases, start=1):
        if case.case_id in previous:
            saved_result = CaseResult(**previous[case.case_id])
            if (
                saved_result.answer.strip()
                and not saved_result.agent_error
                and saved_result.judge_error
            ):
                judge_started = time.perf_counter()
                try:
                    saved_result.judge = judge(
                        case.question,
                        case.reference,
                        saved_result.answer,
                        saved_result.context,
                    )
                    saved_result.judge_error = ""
                except Exception as exc:
                    saved_result.judge = {}
                    saved_result.judge_error = f"{type(exc).__name__}: {exc}"
                saved_result.judge_latency_s = time.perf_counter() - judge_started
                results.append(saved_result)
                report = _report(
                    cases,
                    results,
                    judge_model,
                    generation_model,
                    selection_seed,
                )
                output_path.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                outcome = "JUDGE_ERROR" if saved_result.judge_error else "ok"
                print(f"[{position}/{len(cases)}] {case.case_id}: {outcome}")
                continue
            results.append(saved_result)
            continue
        agent_started = time.perf_counter()
        try:
            agent_result = runner(case.question)
        except Exception as exc:
            result = CaseResult(
                case_id=case.case_id,
                question=case.question,
                reference=case.reference,
                category=case.category,
                answer="",
                context=[],
                tool_names=[],
                tool_errors=[],
                retry_count=0,
                status="agent_error",
                validation_verdicts=[],
                trace_id="",
                agent_latency_s=time.perf_counter() - agent_started,
                judge_latency_s=0.0,
                token_f1=0.0,
                judge={},
                agent_error=f"{type(exc).__name__}: {exc}",
            )
        else:
            answer = _content_text(agent_result.get("answer", ""))
            context = [_content_text(item) for item in agent_result.get("context", [])]
            agent_latency = time.perf_counter() - agent_started
            if not answer.strip():
                result = CaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    reference=case.reference,
                    category=case.category,
                    answer="",
                    context=context,
                    tool_names=list(agent_result.get("tool_names", [])),
                    tool_errors=list(agent_result.get("tool_errors", [])),
                    retry_count=int(agent_result.get("retry_count", 0)),
                    status=str(agent_result.get("status", "empty_answer")),
                    validation_verdicts=list(
                        agent_result.get("validation_verdicts", [])
                    ),
                    trace_id=str(agent_result.get("trace_id", "")),
                    agent_latency_s=agent_latency,
                    judge_latency_s=0.0,
                    token_f1=0.0,
                    judge={},
                    agent_error="EmptyFinalAnswer: Agent returned no final answer",
                )
                results.append(result)
                report = _report(
                    cases,
                    results,
                    judge_model,
                    generation_model,
                    selection_seed,
                )
                output_path.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[{position}/{len(cases)}] {case.case_id}: AGENT_ERROR")
                continue
            judge_started = time.perf_counter()
            try:
                judgment = judge(case.question, case.reference, answer, context)
                judge_error = ""
            except Exception as exc:  # retain a paid Agent run if only judging fails
                judgment = {}
                judge_error = f"{type(exc).__name__}: {exc}"
            result = CaseResult(
                case_id=case.case_id,
                question=case.question,
                reference=case.reference,
                category=case.category,
                answer=answer,
                context=context,
                tool_names=list(agent_result.get("tool_names", [])),
                tool_errors=list(agent_result.get("tool_errors", [])),
                retry_count=int(agent_result.get("retry_count", 0)),
                status=str(agent_result.get("status", "unknown")),
                validation_verdicts=list(
                    agent_result.get("validation_verdicts", [])
                ),
                trace_id=str(agent_result.get("trace_id", "")),
                agent_latency_s=agent_latency,
                judge_latency_s=time.perf_counter() - judge_started,
                token_f1=token_f1(answer, case.reference),
                judge=judgment,
                judge_error=judge_error,
            )
        results.append(result)
        report = _report(
            cases,
            results,
            judge_model,
            generation_model,
            selection_seed,
        )
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        outcome = (
            "AGENT_ERROR"
            if result.agent_error
            else "JUDGE_ERROR"
            if result.judge_error
            else "ok"
        )
        print(f"[{position}/{len(cases)}] {case.case_id}: {outcome}")
    final_report = _report(
        cases,
        results,
        judge_model,
        generation_model,
        selection_seed,
    )
    output_path.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    return final_report


def _dataset_fingerprint(cases: list[EvalCase]) -> str:
    return hashlib.sha256(
        "\n".join(f"{case.case_id}:{case.question}:{case.reference}" for case in cases).encode()
    ).hexdigest()[:16]


def _report(
    cases: list[EvalCase],
    results: list[CaseResult],
    judge_model: str,
    generation_model: str,
    selection_seed: int,
) -> dict[str, Any]:
    return {
        "evaluation_type": "end_to_end_agent_answer_evaluation",
        "note": (
            "Automated source-reference evaluation; this is not clinical "
            "validation and requires human review before deployment claims."
        ),
        "dataset": {
            "source": "final/medical_generalization.csv",
            "n_cases": len(cases),
            "selection_seed": selection_seed,
            "fingerprint": _dataset_fingerprint(cases),
            "case_ids": [case.case_id for case in cases],
        },
        "generation_model": generation_model,
        "judge_model": judge_model,
        "judge_independent_from_generator": judge_model != generation_model,
        "langsmith_tracing_enabled": (
            os.getenv("MEDICAL_RAG_LANGSMITH_TRACING", "false").lower() == "true"
        ),
        "score_scale": "Judge scores normalized from 1-5 to 0-1",
        "metrics": summarize(results),
        "cases": [asdict(result) for result in results],
    }


def _print_summary(report: dict[str, Any], output: Path) -> None:
    metrics = report["metrics"]
    print("\nEnd-to-end Agent Evaluation")
    print("=" * 48)
    for key, value in metrics.items():
        if not key.endswith("_95ci"):
            print(f"{key:<34} {value:.3f}" if isinstance(value, float) else f"{key:<34} {value}")
    print(f"\nDetailed traces: {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="reuse Agent answers but replace judgments with the configured judge model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen case selection without calling external services",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset, limit=args.limit, seed=args.seed)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "n_cases": len(cases),
                    "seed": args.seed,
                    "fingerprint": _dataset_fingerprint(cases),
                    "case_ids": [case.case_id for case in cases],
                },
                indent=2,
            )
        )
        return

    from graphrag.config import GEMINI_MODEL

    judge_model = args.judge_model or os.getenv("EVAL_JUDGE_MODEL") or GEMINI_MODEL
    runner = build_langgraph_runner()
    judge = build_gemini_judge(judge_model)
    report = run_evaluation(
        cases,
        runner,
        judge,
        args.output,
        resume=not args.no_resume,
        judge_model=judge_model,
        generation_model=GEMINI_MODEL,
        selection_seed=args.seed,
        allow_rejudge=args.rejudge,
    )
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
