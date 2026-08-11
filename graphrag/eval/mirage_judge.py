"""Checkpointed evidence and generation judging for scaled Medical MIRAGE runs.

One judge request evaluates the shared top-8 retrieval and both matched-corpus
generations for a case. This avoids paying twice to judge identical evidence.
Primary exact-choice accuracy remains deterministic and is never replaced by
the LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from graphrag.eval.mirage_benchmark import (
    DATASET_ORDER,
    DEFAULT_CACHE,
    MODEL_PRICING_USD_PER_MILLION,
    MirageCase,
    SystemResult,
    _content_text,
    _format_textbook_context,
    _sha256,
    _usage,
    dataset_fingerprint,
    load_benchmark,
    stratified_sample,
)


JUDGE_SCHEMA_VERSION = 1
DEFAULT_SCALED_REPORT = (
    Path(__file__).parent / "external" / "mirage" / "textbooks_scaled_results.json"
)
DEFAULT_JUDGE_OUTPUT = (
    Path(__file__).parent / "external" / "mirage" / "textbooks_scaled_judgments.json"
)
JUDGED_SYSTEMS = ("textbooks-rag", "textbooks-agent")
FAILURE_CATEGORIES = {
    "none",
    "knowledge_missing",
    "outdated_corpus",
    "retrieval_failure",
    "generation_failure",
    "evaluator_failure",
}
CONFLICT_OUTCOMES = {"none", "handled", "failed"}
JUDGE_MAX_OUTPUT_TOKENS = 2048
JUDGE_THINKING_BUDGET = 1024
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "retrieval": {
            "type": "object",
            "properties": {
                "evidence_sufficient": {"type": "boolean"},
                "required_facts_covered": {"type": "integer", "minimum": 0},
                "required_facts_total": {"type": "integer", "minimum": 1},
                "relevant_context_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                },
                "outdated_corpus": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "evidence_sufficient",
                "required_facts_covered",
                "required_facts_total",
                "relevant_context_count",
                "outdated_corpus",
                "reason",
            ],
        },
        "generations": {
            "type": "object",
            "properties": {
                system: {
                    "type": "object",
                    "properties": {
                        "faithful": {"type": "boolean"},
                        "unsupported_claim_count": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "evidence_conflict": {
                            "type": "string",
                            "enum": sorted(CONFLICT_OUTCOMES),
                        },
                        "failure_category": {
                            "type": "string",
                            "enum": sorted(FAILURE_CATEGORIES),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "faithful",
                        "unsupported_claim_count",
                        "evidence_conflict",
                        "failure_category",
                        "reason",
                    ],
                }
                for system in JUDGED_SYSTEMS
            },
            "required": list(JUDGED_SYSTEMS),
        },
    },
    "required": ["retrieval", "generations"],
}


@dataclass(frozen=True)
class JudgeResponse:
    payload: Mapping[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


JudgeRunner = Callable[
    [MirageCase, str, Mapping[str, SystemResult]], JudgeResponse
]


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Judge did not return a JSON object")


def _validated_payload(
    payload: Mapping[str, Any], context_count: int
) -> dict[str, Any]:
    retrieval = payload.get("retrieval")
    generations = payload.get("generations")
    if not isinstance(retrieval, Mapping) or not isinstance(generations, Mapping):
        raise ValueError("Judge payload requires retrieval and generations objects")

    covered = int(retrieval.get("required_facts_covered", -1))
    required = int(retrieval.get("required_facts_total", -1))
    relevant = int(retrieval.get("relevant_context_count", -1))
    if required <= 0 or covered < 0 or covered > required:
        raise ValueError("Invalid required-fact coverage")
    if relevant < 0 or relevant > context_count:
        raise ValueError("Invalid relevant-context count")
    normalized_retrieval = {
        "evidence_sufficient": bool(retrieval.get("evidence_sufficient")),
        "required_facts_covered": covered,
        "required_facts_total": required,
        "judged_recall_at_8": covered / required,
        "relevant_context_count": relevant,
        "outdated_corpus": bool(retrieval.get("outdated_corpus")),
        "reason": str(retrieval.get("reason", "")).strip(),
    }

    normalized_generations: dict[str, Any] = {}
    for system in JUDGED_SYSTEMS:
        item = generations.get(system)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing generation judgment for {system}")
        unsupported = int(item.get("unsupported_claim_count", -1))
        conflict = str(item.get("evidence_conflict", "")).strip()
        category = str(item.get("failure_category", "")).strip()
        if unsupported < 0:
            raise ValueError(f"Invalid unsupported-claim count for {system}")
        if conflict not in CONFLICT_OUTCOMES:
            raise ValueError(f"Invalid evidence-conflict outcome for {system}")
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"Invalid failure category for {system}")
        normalized_generations[system] = {
            "faithful": bool(item.get("faithful")),
            "unsupported_claim_count": unsupported,
            "evidence_conflict": conflict,
            "failure_category": category,
            "reason": str(item.get("reason", "")).strip(),
        }
    return {
        "retrieval": normalized_retrieval,
        "generations": normalized_generations,
    }


def _normalize_failure_categories(
    payload: dict[str, Any], outputs: Mapping[str, SystemResult]
) -> None:
    """Ensure every incorrect answer has one deterministic root-cause slice."""
    retrieval = payload["retrieval"]
    for system in JUDGED_SYSTEMS:
        result = outputs[system]
        generation = payload["generations"][system]
        if result.error:
            generation["failure_category"] = "generation_failure"
            continue
        if result.correct:
            generation["failure_category"] = "none"
            continue
        if retrieval["outdated_corpus"]:
            generation["failure_category"] = "outdated_corpus"
            continue
        if generation["failure_category"] != "none":
            continue
        if retrieval["evidence_sufficient"]:
            category = "generation_failure"
        elif retrieval["relevant_context_count"] == 0:
            category = "retrieval_failure"
        else:
            category = "knowledge_missing"
        generation["failure_category"] = category
        generation["reason"] = (
            generation["reason"]
            + " Deterministic fallback category applied because the judge returned "
            "none for an incorrect exact-choice result."
        ).strip()


def _judge_prompt(
    case: MirageCase,
    context: str,
    outputs: Mapping[str, SystemResult],
) -> str:
    options = "\n".join(f"{key}. {value}" for key, value in case.options.items())
    return f"""You are an independent evaluator for a medical QA benchmark.
Judge evidence coverage and answer grounding; do not solve a different question.
The gold answer is supplied only to assess whether the retrieved evidence covers
the facts needed to justify it. A claim is unsupported when it is not entailed by
the retrieved documents. Treat a justified statement that evidence is missing as
faithful, even though it may be invalid for an exact-choice benchmark.

Return exactly one JSON object with this shape:
{{
  "retrieval": {{
    "evidence_sufficient": true,
    "required_facts_covered": 1,
    "required_facts_total": 1,
    "relevant_context_count": 1,
    "outdated_corpus": false,
    "reason": "brief rationale"
  }},
  "generations": {{
    "textbooks-rag": {{
      "faithful": true,
      "unsupported_claim_count": 0,
      "evidence_conflict": "none",
      "failure_category": "none",
      "reason": "brief rationale"
    }},
    "textbooks-agent": {{
      "faithful": true,
      "unsupported_claim_count": 0,
      "evidence_conflict": "none",
      "failure_category": "none",
      "reason": "brief rationale"
    }}
  }}
}}

Definitions:
- evidence_sufficient: top-8 context alone supports choosing the gold option.
- required_facts coverage: count atomic facts needed for the gold answer and how
  many are present; this ratio is judged Recall@8.
- relevant_context_count: documents that materially help answer the question.
- outdated_corpus: missing evidence is specifically due to knowledge newer than
  the textbook corpus, not merely a narrow topic omission.
- evidence_conflict: none when documents do not conflict; handled when the answer
  resolves a real conflict correctly; failed when it resolves one incorrectly.
- failure_category: for a wrong/invalid answer choose the single primary root
  cause from knowledge_missing, outdated_corpus, retrieval_failure,
  generation_failure, evaluator_failure. Use none for a correct answer. Use
  evaluator_failure only when the saved output clearly contains a valid final
  choice that the deterministic parser missed.

QUESTION:
{case.retrieval_query}

OPTIONS:
{options}

GOLD ANSWER:
{case.answer}. {case.options[case.answer]}

TOP-8 TEXTBOOK CONTEXT:
{context}

TEXTBOOKS-RAG OUTPUT:
Status: {outputs['textbooks-rag'].status}
Provider error: {outputs['textbooks-rag'].error or 'none'}
{outputs['textbooks-rag'].raw_answer}

TEXTBOOKS-AGENT OUTPUT:
Status: {outputs['textbooks-agent'].status}
Provider error: {outputs['textbooks-agent'].error or 'none'}
{outputs['textbooks-agent'].raw_answer}
"""


def build_judge_runner(model: str) -> JudgeRunner:
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from graphrag.config import (
        GEMINI_MAX_RETRIES,
        GEMINI_REQUEST_TIMEOUT_S,
        GOOGLE_API_KEY,
    )

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        request_timeout=GEMINI_REQUEST_TIMEOUT_S,
        retries=GEMINI_MAX_RETRIES,
        max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        thinking_budget=JUDGE_THINKING_BUDGET,
        response_mime_type="application/json",
        response_schema=JUDGE_RESPONSE_SCHEMA,
    )

    def run(
        case: MirageCase,
        context: str,
        outputs: Mapping[str, SystemResult],
    ) -> JudgeResponse:
        response = llm.invoke(
            [HumanMessage(content=_judge_prompt(case, context, outputs))]
        )
        usage = _usage(response)
        return JudgeResponse(
            payload=_extract_json(_content_text(response.content)),
            **usage,
        )

    return run


def _generation_results(report: Mapping[str, Any]) -> dict[str, dict[str, SystemResult]]:
    by_case: dict[str, dict[str, SystemResult]] = {}
    for item in report.get("results", []):
        result = SystemResult(**item)
        if result.system in JUDGED_SYSTEMS:
            by_case.setdefault(result.case_id, {})[result.system] = result
    return by_case


def _summarize(
    generation_report: Mapping[str, Any],
    judgments: list[Mapping[str, Any]],
    pricing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    successful = [item for item in judgments if not item.get("error")]
    retrieval = [item["retrieval"] for item in successful]
    input_tokens = sum(int(item.get("input_tokens", 0)) for item in judgments)
    output_tokens = sum(int(item.get("output_tokens", 0)) for item in judgments)
    generation_metrics: dict[str, Any] = {}
    failure_by_category: dict[str, Any] = {
        system: {} for system in JUDGED_SYSTEMS
    }
    result_by_case = _generation_results(generation_report)
    failure_by_dataset: dict[str, Any] = {
        system: {} for system in JUDGED_SYSTEMS
    }

    for system in JUDGED_SYSTEMS:
        evaluated = [item["generations"][system] for item in successful]
        sufficient_results = [
            result_by_case[str(item["case_id"])][system]
            for item in successful
            if item["retrieval"]["evidence_sufficient"]
        ]
        insufficient_results = [
            result_by_case[str(item["case_id"])][system]
            for item in successful
            if not item["retrieval"]["evidence_sufficient"]
        ]
        conflicts = [
            item for item in evaluated if item["evidence_conflict"] != "none"
        ]
        generation_metrics[system] = {
            "n_judged": len(evaluated),
            "faithfulness_rate": statistics.fmean(
                float(item["faithful"]) for item in evaluated
            )
            if evaluated
            else None,
            "unsupported_claim_rate": statistics.fmean(
                float(item["unsupported_claim_count"] > 0) for item in evaluated
            )
            if evaluated
            else None,
            "average_unsupported_claims": statistics.fmean(
                item["unsupported_claim_count"] for item in evaluated
            )
            if evaluated
            else None,
            "evidence_conflict_handling_rate": statistics.fmean(
                float(item["evidence_conflict"] == "handled") for item in conflicts
            )
            if conflicts
            else None,
            "accuracy_when_evidence_sufficient": statistics.fmean(
                float(item.correct) for item in sufficient_results
            )
            if sufficient_results
            else None,
            "accuracy_when_evidence_insufficient": statistics.fmean(
                float(item.correct) for item in insufficient_results
            )
            if insufficient_results
            else None,
        }
        for item in successful:
            case_id = str(item["case_id"])
            result = result_by_case.get(case_id, {}).get(system)
            if result is None or result.correct:
                continue
            category = item["generations"][system]["failure_category"]
            failure_by_category[system].setdefault(category, []).append(case_id)

        for dataset in DATASET_ORDER:
            subset = [
                values[system]
                for values in result_by_case.values()
                if system in values and values[system].dataset == dataset
            ]
            if subset:
                failure_by_dataset[system][dataset] = {
                    "n": len(subset),
                    "accuracy": sum(item.correct for item in subset) / len(subset),
                    "incorrect_case_ids": [
                        item.case_id for item in subset if not item.correct
                    ],
                }

    for item in judgments:
        if not item.get("error"):
            continue
        for system in JUDGED_SYSTEMS:
            failure_by_category[system].setdefault(
                "evaluator_failure", []
            ).append(
                str(item["case_id"])
            )

    total_contexts = sum(int(item.get("context_count", 0)) for item in successful)
    relevant_contexts = sum(item["relevant_context_count"] for item in retrieval)
    return {
        "retrieval": {
            "n_judged": len(retrieval),
            "evidence_sufficiency_rate": statistics.fmean(
                float(item["evidence_sufficient"]) for item in retrieval
            )
            if retrieval
            else None,
            "judged_recall_at_8": statistics.fmean(
                item["judged_recall_at_8"] for item in retrieval
            )
            if retrieval
            else None,
            "relevant_context_rate": relevant_contexts / total_contexts
            if total_contexts
            else None,
            "outdated_corpus_rate": statistics.fmean(
                float(item["outdated_corpus"]) for item in retrieval
            )
            if retrieval
            else None,
        },
        "generation": generation_metrics,
        "failure_slices": {
            "by_dataset": failure_by_dataset,
            "by_category": failure_by_category,
        },
        "judge_system": {
            "n": len(judgments),
            "error_rate": (
                sum(bool(item.get("error")) for item in judgments) / len(judgments)
                if judgments
                else 0.0
            ),
            "p50_latency_s": _percentile(
                [float(item.get("latency_s", 0.0)) for item in judgments], 0.50
            ),
            "p95_latency_s": _percentile(
                [float(item.get("latency_s", 0.0)) for item in judgments], 0.95
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": (
                input_tokens * float(pricing["input"])
                + output_tokens * float(pricing["output_including_thinking"])
            )
            / 1_000_000
            if pricing
            else None,
        },
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(quantile * len(ordered) - 1e-9)))
    return ordered[position]


def run_judging(
    generation_report_path: Path,
    dataset_path: Path,
    index_path: Path,
    output_path: Path,
    judge_model: str,
    runner: JudgeRunner | None = None,
    resume: bool = True,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    from graphrag.eval.mirage_corpus import TextbooksBM25Retriever

    generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    benchmark = generation_report.get("benchmark", {})
    if benchmark.get("source_sha256") != _sha256(dataset_path):
        raise ValueError("Generation report uses a different benchmark source")
    if not benchmark.get("question_only_retrieval"):
        raise ValueError("Generation report did not enforce question-only retrieval")
    systems = generation_report.get("systems", [])
    if not all(system in systems for system in JUDGED_SYSTEMS):
        raise ValueError("Generation report lacks both matched-corpus systems")

    cases = stratified_sample(
        load_benchmark(
            dataset_path, enforce_official_counts=enforce_official_counts
        ),
        int(benchmark["n_selected"]),
        int(benchmark["selection_seed"]),
    )
    if dataset_fingerprint(cases) != benchmark.get("selection_fingerprint"):
        raise ValueError("Generation report selection fingerprint is invalid")
    results = _generation_results(generation_report)
    if any(set(results.get(case.case_id, {})) != set(JUDGED_SYSTEMS) for case in cases):
        raise ValueError("Generation report is incomplete for judged systems")

    previous: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if saved.get("generation_selection_fingerprint") != dataset_fingerprint(cases):
            raise ValueError("Judge checkpoint uses a different generation selection")
        if saved.get("judge_model") != judge_model:
            raise ValueError("Judge checkpoint uses a different model")
        if saved.get("judge_schema_version") != JUDGE_SCHEMA_VERSION:
            raise ValueError("Judge checkpoint uses a different schema")
        previous = {
            str(item["case_id"]): item
            for item in saved.get("judgments", [])
            if not item.get("error")
        }
        for case_id, item in previous.items():
            _normalize_failure_categories(item, results[case_id])

    retriever = TextbooksBM25Retriever(index_path)
    effective_runner = runner or build_judge_runner(judge_model)
    judgments: list[dict[str, Any]] = []
    pricing = MODEL_PRICING_USD_PER_MILLION.get(judge_model)

    for position, case in enumerate(cases, start=1):
        if case.case_id in previous:
            judgments.append(previous[case.case_id])
            continue
        case_results = results[case.case_id]
        snippets = retriever.retrieve(case.retrieval_query, k=8)
        retrieved_ids = [snippet.snippet_id for snippet in snippets]
        for system in JUDGED_SYSTEMS:
            if case_results[system].error:
                continue
            if case_results[system].retrieved_ids != retrieved_ids:
                raise ValueError(
                    f"Retrieved IDs changed for {case.case_id} ({system})"
                )
        context = _format_textbook_context(snippets)
        started = time.perf_counter()
        try:
            response = effective_runner(case, context, case_results)
            payload = _validated_payload(response.payload, len(snippets))
            for system in JUDGED_SYSTEMS:
                if not case_results[system].error:
                    continue
                payload["generations"][system] = {
                    "faithful": False,
                    "unsupported_claim_count": 0,
                    "evidence_conflict": "none",
                    "failure_category": "generation_failure",
                    "reason": (
                        "No answer was produced because the provider request failed: "
                        + case_results[system].error
                    ),
                }
            _normalize_failure_categories(payload, case_results)
            error = ""
        except Exception as exc:
            response = JudgeResponse(payload={})
            payload = {}
            error = f"{type(exc).__name__}: {exc}"
        judgment = {
            "case_id": case.case_id,
            "dataset": case.dataset,
            "context_count": len(snippets),
            **payload,
            "latency_s": time.perf_counter() - started,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "error": error,
        }
        judgments.append(judgment)
        report = {
            "evaluation_type": "external_medical_mirage_evidence_judge",
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
            "judge_model": judge_model,
            "generation_report": str(generation_report_path),
            "generation_selection_fingerprint": dataset_fingerprint(cases),
            "pricing_usd_per_million_tokens": dict(pricing) if pricing else None,
            "metrics": _summarize(generation_report, judgments, pricing),
            "judgments": judgments,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{position}/{len(cases)}] {case.case_id}: "
            f"{'ERROR' if error else 'judged'}"
        )

    final = {
        "evaluation_type": "external_medical_mirage_evidence_judge",
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "judge_model": judge_model,
        "generation_report": str(generation_report_path),
        "generation_selection_fingerprint": dataset_fingerprint(cases),
        "pricing_usd_per_million_tokens": dict(pricing) if pricing else None,
        "metrics": _summarize(generation_report, judgments, pricing),
        "judgments": judgments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def main(argv: list[str] | None = None) -> None:
    from graphrag.eval.mirage_corpus import DEFAULT_INDEX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-report", type=Path, default=DEFAULT_SCALED_REPORT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--textbooks-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGE_OUTPUT)
    parser.add_argument(
        "--model",
        default=os.getenv("MIRAGE_JUDGE_MODEL", "gemini-2.5-flash"),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    report = run_judging(
        args.generation_report,
        args.dataset,
        args.textbooks_index,
        args.output,
        args.model,
        resume=not args.no_resume,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
