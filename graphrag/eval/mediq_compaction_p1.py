"""Frozen guarded runner for the Phase 1 compaction main experiment.

Executes the pre-committed spec `specs/mediq_compaction_p1.json`: a new
untouched MediQ holdout (seed 47, 20 cases per specialty, n=100, all Stage
5N/5O development IDs excluded) with five matched diagnostic arms. The
pre-registered primary comparison is structured-handoff vs full-transcript
(paired exact McNemar); everything else is secondary.

Guards:
- The frozen selection is re-derived from the spec and must match exactly.
- Paid execution requires --execute after reviewing --dry-run.
- Hard runtime guards: max provider calls, max prompt chars, and a cost guard
  (default $4) enforced call-by-call by CheckpointedGeminiProvider.
- Note: unlike the Stage 5O runner, the pre-check compares the guard against
  the Stage 5O-extrapolated expected cost (~$1.5-2.5), not the theoretical
  worst-case bound (which prices every call at the prompt-char ceiling and
  would exceed any reasonable guard at ~1,300 max calls). The runtime cost
  guard remains hard and conservative.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from graphrag.eval.checkpointed_gemini import CheckpointedGeminiProvider
from graphrag.eval.mediq_handoff_benchmark import (
    MAX_PROMPT_CHARS,
    _case_run,
    _save_json,
    summarize,
)
from graphrag.eval.mediq_handoff_data import (
    DEFAULT_ROOT,
    DEFAULT_SOURCE,
    ConceptAwareFactPatient,
    CoverageAwareQuestionRefiner,
    MediQCase,
    canonical_hash,
    clinical_tool_fingerprint,
    sha256,
)
from graphrag.eval.mediq_handoff_holdout import _strip_fact_index
from graphrag.eval.mirage_corpus import DEFAULT_INDEX, TextbooksBM25Retriever

P1_SPEC = Path(__file__).parent / "specs" / "mediq_compaction_p1.json"
P1_CHECKPOINT = DEFAULT_ROOT / "p1_provider_checkpoint.json"
P1_OUTPUT = DEFAULT_ROOT / "p1_compaction_results.json"
P1_CONDITIONS = (
    "full-transcript",
    "truncation-headtail",
    "freetext-summary",
    "structured-handoff",
    "handoff-plus-sources",
)
PRIMARY_CONDITION = "structured-handoff"
MAX_P1_PROVIDER_CALLS = 1_300
DEFAULT_P1_COST_GUARD_USD = 4.0
P1_MAX_OUTPUT_TOKENS = 2_048
STAGE5O_COST_PER_CASE_3ARM_USD = 0.33 / 30


def load_p1_spec(path: Path = P1_SPEC) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if (
        spec.get("stage") != "P1"
        or spec.get("split_role") != "untouched_holdout"
        or tuple(spec.get("diagnostic_conditions", ())) != P1_CONDITIONS
        or int(spec.get("cases_per_specialty", 0)) != 20
    ):
        raise ValueError("Unsupported Phase 1 compaction spec")
    return spec


def load_p1_cases(
    source_path: Path = DEFAULT_SOURCE, spec: Mapping[str, Any] | None = None
) -> list[MediQCase]:
    spec = dict(spec or load_p1_spec())
    if sha256(source_path) != spec["source_sha256"]:
        raise ValueError("MediQ source SHA-256 does not match the P1 spec")
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    excluded = {int(value) for value in spec["excluded_source_ids"]}
    rng = random.Random(int(spec["seed"]))
    selected: list[Mapping[str, Any]] = []
    for specialty in spec["specialties"]:
        candidates = [
            row
            for row in rows
            if row.get("patient", {}).get("gpt_specialty") == specialty
            and int(row["id"]) not in excluded
        ]
        selected.extend(rng.sample(candidates, int(spec["cases_per_specialty"])))
    selected_ids = [int(row["id"]) for row in selected]
    if selected_ids != spec["selected_source_ids"]:
        raise ValueError("Frozen P1 selection changed; refusing to run")

    cases: list[MediQCase] = []
    for row in selected:
        context = tuple(
            str(value).strip() for value in row["context"] if str(value).strip()
        )
        facts = tuple(
            _strip_fact_index(value)
            for value in row.get("facts", [])
            if _strip_fact_index(value)
        )
        options = {str(key): str(value) for key, value in row["options"].items()}
        answer_choice = str(row["answer_idx"]).upper()
        if not context or not facts or answer_choice not in options:
            raise ValueError(f"Invalid selected MediQ case: {row.get('id')}")
        cases.append(
            MediQCase(
                case_id=f"mediq-{row['id']}",
                source_id=int(row["id"]),
                specialty=str(row["patient"]["gpt_specialty"]),
                question=str(row["question"]).strip(),
                initial_info=context[0],
                context=context,
                facts=facts,
                options=options,
                answer_choice=answer_choice,
            )
        )
    return cases


def p1_fingerprint(cases: list[MediQCase], spec: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "stage": "P1",
            "source_sha256": spec["source_sha256"],
            "seed": spec["seed"],
            "ids": [case.source_id for case in cases],
            "patient_matcher": spec["patient_matcher"],
            "question_refiner": spec["question_refiner"],
            "clinical_tool_fingerprint": spec["clinical_tool_fingerprint"],
            "conditions": list(P1_CONDITIONS),
        }
    )


def p1_dry_run_plan(
    cases: list[MediQCase], spec: Mapping[str, Any], model: str
) -> dict[str, Any]:
    max_interviewer_calls = len(cases) * (int(spec["max_questions"]) + 1)
    summary_calls = len(cases)
    diagnosis_calls = len(cases) * len(P1_CONDITIONS)
    expected_calls = max_interviewer_calls + summary_calls + diagnosis_calls
    return {
        "stage": "P1",
        "split_role": "untouched_holdout",
        "dataset_fingerprint": p1_fingerprint(cases, spec),
        "cases": len(cases),
        "diagnostic_conditions": list(P1_CONDITIONS),
        "primary_comparison": f"{PRIMARY_CONDITION} vs full-transcript",
        "max_interviewer_calls": max_interviewer_calls,
        "summary_calls": summary_calls,
        "diagnosis_calls": diagnosis_calls,
        "expected_logical_calls": expected_calls,
        "max_provider_calls": MAX_P1_PROVIDER_CALLS,
        "patient_matcher": ConceptAwareFactPatient.matcher_version,
        "question_refiner": CoverageAwareQuestionRefiner.version,
        "clinical_tool_fingerprint": clinical_tool_fingerprint(),
        "expected_cost_extrapolated_from_stage5o_usd": round(
            len(cases) * STAGE5O_COST_PER_CASE_3ARM_USD * (10 / 7), 2
        ),
        "configured_cost_guard_usd": DEFAULT_P1_COST_GUARD_USD,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_output_tokens": P1_MAX_OUTPUT_TOKENS,
        "requires_explicit_execute_flag": True,
    }


def run_p1(
    *,
    cases: list[MediQCase],
    spec: Mapping[str, Any],
    model: str,
    index_path: Path,
    provider_checkpoint: Path,
    output_path: Path,
    cost_guard_usd: float = DEFAULT_P1_COST_GUARD_USD,
    reuse_only: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    fingerprint = p1_fingerprint(cases, spec)
    provider = CheckpointedGeminiProvider(
        model=model,
        checkpoint_path=provider_checkpoint,
        fingerprint=fingerprint,
        max_calls=MAX_P1_PROVIDER_CALLS,
        max_cost_usd=cost_guard_usd,
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_output_tokens=P1_MAX_OUTPUT_TOKENS,
        allow_new_calls=not reuse_only,
    )
    retriever = TextbooksBM25Retriever(index_path)
    run_cases = cases[:limit] if limit else cases
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    log_path = output_path.with_name("p1_run_log.jsonl")
    started_at = time.time()
    _append_jsonl(
        log_path,
        {
            "event": "run_start",
            "ts": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "model": model,
            "cases": len(run_cases),
            "conditions": list(P1_CONDITIONS),
            "cost_guard_usd": cost_guard_usd,
            "reuse_only": reuse_only,
            "cost_already_checkpointed_usd": round(provider.spent, 6),
        },
    )
    for position, case in enumerate(run_cases, start=1):
        try:
            result = _case_run(
                case,
                provider=provider,
                retriever=retriever,
                max_questions=int(spec["max_questions"]),
                top_k=int(spec["retrieval"]["top_k"]),
                diagnostic_conditions=P1_CONDITIONS,
                patient_factory=ConceptAwareFactPatient,
                question_refiner=CoverageAwareQuestionRefiner(),
            )
        except (RuntimeError, KeyboardInterrupt) as error:
            # Guard trips (call/cost ceiling) and operator interrupts are fatal
            # by design: stop, keep whatever is already saved, do not retry.
            _append_jsonl(
                log_path,
                {
                    "event": "run_halted",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "position": position,
                    "case_id": case.case_id,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "cases_completed": len(results),
                    "spent_usd": round(provider.spent, 6),
                },
            )
            print(f"[{position}/{len(run_cases)}] {case.case_id}: HALTED {error}", flush=True)
            raise
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            # A single malformed case must not destroy a paid run. The failure
            # is recorded with its type and message so it stays auditable, and
            # metrics are computed only over completed cases.
            failures.append(
                {
                    "case_id": case.case_id,
                    "source_id": case.source_id,
                    "specialty": case.specialty,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
            print(
                f"[{position}/{len(run_cases)}] {case.case_id}: FAILED "
                f"{type(error).__name__}: {str(error)[:200]}",
                flush=True,
            )
            _append_jsonl(
                log_path,
                {
                    "event": "case_failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "position": position,
                    "case_id": case.case_id,
                    "specialty": case.specialty,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "spent_usd": round(provider.spent, 6),
                },
            )
            if report:
                report["failed_cases"] = list(failures)
                _save_json(output_path, report)
            continue
        results.append(result)
        report = {
            "format_version": 1,
            "stage": "P1",
            "benchmark": "MediQ compaction main experiment",
            "dataset_fingerprint": fingerprint,
            "source_revision": spec["revision"],
            "source_sha256": spec["source_sha256"],
            "model": model,
            "cost_guard_usd": cost_guard_usd,
            "primary_condition": PRIMARY_CONDITION,
            "failed_cases": list(failures),
            "results": results,
            "metrics": summarize(
                results,
                provider,
                diagnostic_conditions=P1_CONDITIONS,
                primary_condition=PRIMARY_CONDITION,
            ),
        }
        _save_json(output_path, report)
        _append_jsonl(
            log_path,
            {
                "event": "case_done",
                "ts": datetime.now(timezone.utc).isoformat(),
                "position": position,
                "case_id": case.case_id,
                "specialty": case.specialty,
                "status": result.get("status"),
                "questions_asked": result.get("questions_asked"),
                "correct_by_condition": {
                    condition: value.get("correct")
                    for condition, value in (result.get("diagnoses") or {}).items()
                },
                "diagnosis_input_tokens": {
                    condition: value.get("input_tokens")
                    for condition, value in (result.get("diagnoses") or {}).items()
                },
                "spent_usd": round(provider.spent, 6),
                "elapsed_s": round(time.time() - started_at, 1),
            },
        )
        print(
            f"[{position}/{len(run_cases)}] {case.case_id}: {result['status']} "
            f"questions={result['questions_asked']} cost=${provider.spent:.4f}",
            flush=True,
        )
    _append_jsonl(
        log_path,
        {
            "event": "run_end",
            "ts": datetime.now(timezone.utc).isoformat(),
            "cases_completed": len(results),
            "cases_failed": len(failures),
            "spent_usd": round(provider.spent, 6),
            "elapsed_s": round(time.time() - started_at, 1),
        },
    )
    return report


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one structured log line. Never raises: observability must not be
    able to kill a paid run. Written after each case so an interrupted run
    still leaves a reconstructable timeline (the earlier P1 abort left none)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 - logging is best-effort by design
        pass


def checkpoint_status(
    checkpoint_path: Path = P1_CHECKPOINT, output_path: Path = P1_OUTPUT
) -> dict[str, Any]:
    """Zero-cost progress report derived from the saved checkpoint."""
    if not checkpoint_path.exists():
        return {"checkpoint_exists": False, "path": str(checkpoint_path)}
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    calls = payload.get("calls", {})
    stages: dict[str, int] = {}
    per_case: dict[str, int] = {}
    for call_id, record in calls.items():
        stage = str(record.get("stage", "?"))
        stages[stage] = stages.get(stage, 0) + 1
        per_case[call_id.split(":", 1)[0]] = per_case.get(call_id.split(":", 1)[0], 0) + 1
    # A fully processed case makes 5 diagnosis calls + 1 summary + >=1 interview.
    complete = [case for case, count in per_case.items() if count >= 7]
    status: dict[str, Any] = {
        "checkpoint_exists": True,
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "model": payload.get("model"),
        "total_calls": len(calls),
        "calls_by_stage": dict(sorted(stages.items())),
        "cases_touched": len(per_case),
        "cases_looking_complete": len(complete),
        "last_case_touched": list(per_case)[-1] if per_case else None,
        "results_file_exists": output_path.exists(),
    }
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        status["cases_in_results"] = len(report.get("results", []))
        status["failed_cases"] = report.get("failed_cases", [])
        metrics = report.get("metrics", {})
        provider = metrics.get("provider", {})
        status["estimated_cost_usd"] = provider.get("estimated_cost_usd")
        status["accuracy_by_condition"] = {
            condition: value.get("accuracy")
            for condition, value in metrics.get("diagnostic_conditions", {}).items()
        }
    log_path = output_path.with_name("p1_run_log.jsonl")
    if log_path.exists():
        lines = [
            line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        status["log_lines"] = len(lines)
        status["log_tail"] = [json.loads(line) for line in lines[-3:]]
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=P1_SPEC)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--provider-checkpoint", type=Path, default=P1_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=P1_OUTPUT)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--cost-guard", type=float, default=DEFAULT_P1_COST_GUARD_USD)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print zero-cost progress from the saved checkpoint and exit",
    )
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N cases (smoke check)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Permit guarded provider calls after the dry-run plan is approved",
    )
    args = parser.parse_args(argv)

    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if args.status:
        print(json.dumps(checkpoint_status(args.provider_checkpoint, args.output), indent=2))
        return
    spec = load_p1_spec(args.spec)
    if not args.source.exists():
        raise FileNotFoundError("Pinned MediQ source is missing")
    cases = load_p1_cases(args.source, spec)
    plan = p1_dry_run_plan(cases, spec, args.model)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.execute and not args.reuse_only:
        raise RuntimeError("Paid P1 run is locked; review --dry-run then pass --execute")
    if plan["expected_cost_extrapolated_from_stage5o_usd"] > args.cost_guard:
        raise RuntimeError("Extrapolated expected cost exceeds the configured guard")
    report = run_p1(
        cases=cases,
        spec=spec,
        model=args.model,
        index_path=args.index,
        provider_checkpoint=args.provider_checkpoint,
        output_path=args.output,
        cost_guard_usd=args.cost_guard,
        reuse_only=args.reuse_only,
        limit=args.limit,
    )
    metrics = report["metrics"]
    slim = {
        "cases_completed": metrics["cases_completed"],
        "accuracy_by_condition": {
            condition: value["accuracy"]
            for condition, value in metrics["diagnostic_conditions"].items()
        },
        "paired": metrics["paired"],
        "estimated_cost_usd": metrics["provider"]["estimated_cost_usd"],
    }
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
