"""Small online smoke suite for the complete LangGraph agent workflow.

This is an execution check, not the final end-to-end quality benchmark. It
verifies bounded tool use, non-empty final answers, reflection state, and a few
case-specific safety/content keywords.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import ToolMessage

from graphrag.agent.react_agent import AgentState, MAX_RETRIES, build_agent


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "eval" / "data" / "agent_smoke_results.json"
)


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    question: str
    expected_any: tuple[str, ...]


SMOKE_CASES = (
    SmokeCase(
        "contraindication",
        "A patient with hepatorenal syndrome and recent cardiac ischemia is being "
        "considered for terlipressin. Is it appropriate?",
        ("contraindicated", "should not", "not appropriate"),
    ),
    SmokeCase(
        "pancreatitis_nutrition",
        "For mild acute pancreatitis, when and what should the patient start eating?",
        ("low-fat", "24", "48"),
    ),
    SmokeCase(
        "difficult_airway",
        "Which first-line laryngoscope blade is preferred for a difficult intubation?",
        ("hyperangulated",),
    ),
    SmokeCase(
        "ptsd_treatment",
        "What is the recommended first-line treatment approach for PTSD?",
        ("psychotherap",),
    ),
    SmokeCase(
        "pregnancy_hiv",
        "Can bictegravir/FTC/TAF be considered for initial HIV therapy during pregnancy?",
        ("alternative", "bictegravir"),
    ),
)


def run_case(graph, case: SmokeCase) -> dict:
    started = time.perf_counter()
    result = graph.invoke(
        AgentState(
            query=case.question,
            messages=[],
            retrieved_context=[],
            retry_count=0,
            final_answer="",
            candidate_answers=[],
            validation_verdicts=[],
            status="running",
        ),
        config={"recursion_limit": 20},
    )
    elapsed = time.perf_counter() - started
    tool_messages = [
        message
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    tool_names = [message.name for message in tool_messages if message.name]
    answer = result.get("final_answer", "")
    keyword_pass = any(term in answer.lower() for term in case.expected_any)
    required_tools_pass = (
        "retrieve_graph" in tool_names and "retrieve_vector" in tool_names
    )
    pipeline_pass = bool(answer.strip()) and required_tools_pass and (
        int(result.get("retry_count", 0)) <= MAX_RETRIES
    )
    return {
        "case_id": case.case_id,
        "question": case.question,
        "answer": answer,
        "tool_names": tool_names,
        "tool_errors": [
            str(message.content)
            for message in tool_messages
            if "Tool unavailable" in str(message.content)
        ],
        "context_count": len(result.get("retrieved_context", [])),
        "retry_count": int(result.get("retry_count", 0)),
        "candidate_answers": result.get("candidate_answers", []),
        "validation_verdicts": result.get("validation_verdicts", []),
        "status": result.get("status", "unknown"),
        "latency_s": elapsed,
        "keyword_pass": keyword_pass,
        "required_tools_pass": required_tools_pass,
        "pipeline_pass": pipeline_pass,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    return {
        "n_cases": count,
        "pipeline_passed": sum(bool(item["pipeline_pass"]) for item in results),
        "keyword_passed": sum(bool(item["keyword_pass"]) for item in results),
        "approved": sum(item["status"] == "approved" for item in results),
        "tool_errors": sum(len(item["tool_errors"]) for item in results),
        "average_retries": (
            sum(item["retry_count"] for item in results) / count if count else 0.0
        ),
        "average_latency_s": (
            sum(item["latency_s"] for item in results) / count if count else 0.0
        ),
    }


def save_report(results: list[dict], output: Path) -> dict:
    report = {
        "evaluation_type": "online_agent_execution_smoke_test",
        "note": "Pipeline smoke test; not a clinical quality evaluation.",
        "summary": summarize(results),
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=len(SMOKE_CASES))
    parser.add_argument(
        "--case-id",
        action="append",
        choices=[case.case_id for case in SMOKE_CASES],
        help="run named smoke cases instead of the leading --limit cases; repeatable",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    cases = (
        [case for case in SMOKE_CASES if case.case_id in args.case_id]
        if args.case_id
        else list(SMOKE_CASES[: args.limit])
    )
    previous: dict[str, dict] = {}
    if not args.no_resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        previous = {item["case_id"]: item for item in saved.get("cases", [])}

    results: list[dict] = []
    graph = None
    for index, case in enumerate(cases, start=1):
        if case.case_id in previous:
            result = previous[case.case_id]
        else:
            graph = graph or build_agent()
            result = run_case(graph, case)
        results.append(result)
        save_report(results, args.output)
        print(
            f"[{index}/{len(cases)}] {case.case_id}: "
            f"pipeline={result['pipeline_pass']} keyword={result['keyword_pass']} "
            f"status={result['status']} retries={result['retry_count']}"
        )

    report = save_report(results, args.output)
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["pipeline_passed"] != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
