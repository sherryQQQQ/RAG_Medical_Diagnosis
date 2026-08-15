"""Frozen zero-call plan and guarded runner for the Stage 5O MediQ holdout."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping

from graphrag.eval.checkpointed_gemini import CheckpointedGeminiProvider
from graphrag.eval.mediq_handoff_benchmark import (
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_CHARS,
    _case_run,
    _save_json,
    dry_run_plan,
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
    download_source,
    sha256,
)
from graphrag.eval.mirage_corpus import DEFAULT_INDEX, TextbooksBM25Retriever


DEFAULT_HOLDOUT_SPEC = Path(__file__).parent / "specs" / "mediq_handoff_holdout.json"
DEFAULT_HOLDOUT_CHECKPOINT = DEFAULT_ROOT / "stage5o_provider_checkpoint.json"
DEFAULT_HOLDOUT_OUTPUT = DEFAULT_ROOT / "stage5o_handoff_results.json"
HOLDOUT_DIAGNOSTIC_CONDITIONS = (
    "full-transcript",
    "structured-handoff",
    "handoff-plus-sources",
)
HOLDOUT_CASES = 30
MAX_HOLDOUT_PROVIDER_CALLS = 210
DEFAULT_HOLDOUT_COST_GUARD_USD = 2.0
STAGE5N_COST_PER_CASE_USD = 0.0501681 / 5


def load_holdout_spec(path: Path = DEFAULT_HOLDOUT_SPEC) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "stage",
        "revision",
        "source_sha256",
        "license",
        "split_role",
        "seed",
        "cases_per_specialty",
        "specialties",
        "excluded_source_ids",
        "selected_source_ids",
        "max_questions",
        "patient_matcher",
        "question_refiner",
        "clinical_tool_fingerprint",
        "retrieval",
        "diagnostic_conditions",
    }
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError("Invalid MediQ Stage 5O holdout spec")
    if (
        spec["format_version"] != 1
        or spec["stage"] != "5O"
        or spec["license"] != "CC-BY-4.0"
        or spec["split_role"] != "untouched_holdout"
    ):
        raise ValueError("Unsupported MediQ Stage 5O holdout spec")
    if spec["diagnostic_conditions"] != list(HOLDOUT_DIAGNOSTIC_CONDITIONS):
        raise ValueError("Unexpected Stage 5O diagnostic conditions")
    if spec["patient_matcher"] != ConceptAwareFactPatient.matcher_version:
        raise ValueError("Stage 5O patient matcher version changed")
    if spec["question_refiner"] != CoverageAwareQuestionRefiner.version:
        raise ValueError("Stage 5O question refiner version changed")
    if spec["clinical_tool_fingerprint"] != clinical_tool_fingerprint():
        raise ValueError("Stage 5O deterministic clinical tools changed")
    if spec["max_questions"] != 3 or spec["retrieval"].get("top_k") != 8:
        raise ValueError("Stage 5O question or retrieval budget changed")
    if len(spec["selected_source_ids"]) != HOLDOUT_CASES:
        raise ValueError("Stage 5O holdout must contain exactly 30 cases")
    if set(spec["selected_source_ids"]) & set(spec["excluded_source_ids"]):
        raise ValueError("Stage 5O holdout overlaps the Stage 5N development set")
    return spec


def _strip_fact_index(text: str) -> str:
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", str(text)).strip()


def load_holdout_cases(
    source_path: Path = DEFAULT_SOURCE,
    spec: Mapping[str, Any] | None = None,
) -> list[MediQCase]:
    spec = dict(spec or load_holdout_spec())
    if sha256(source_path) != spec["source_sha256"]:
        raise ValueError("MediQ source SHA-256 does not match the holdout spec")
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
        raise ValueError(f"Frozen Stage 5O selection changed: {selected_ids}")

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


def holdout_fingerprint(
    cases: list[MediQCase], spec: Mapping[str, Any]
) -> str:
    return canonical_hash(
        {
            "stage": "5O",
            "source_sha256": spec["source_sha256"],
            "seed": spec["seed"],
            "ids": [case.source_id for case in cases],
            "patient_matcher": spec["patient_matcher"],
            "question_refiner": spec["question_refiner"],
            "clinical_tool_fingerprint": spec["clinical_tool_fingerprint"],
            "conditions": list(HOLDOUT_DIAGNOSTIC_CONDITIONS),
        }
    )


def holdout_dry_run_plan(
    cases: list[MediQCase], spec: Mapping[str, Any], model: str
) -> dict[str, Any]:
    plan = dry_run_plan(
        cases,
        spec,
        model,
        diagnostic_conditions=HOLDOUT_DIAGNOSTIC_CONDITIONS,
    )
    plan.update(
        {
            "stage": "5O",
            "split_role": "untouched_holdout",
            "dataset_fingerprint": holdout_fingerprint(cases, spec),
            "patient_matcher": ConceptAwareFactPatient.matcher_version,
            "question_refiner": CoverageAwareQuestionRefiner.version,
            "clinical_tool_fingerprint": clinical_tool_fingerprint(),
            "expected_cost_extrapolated_from_stage5n_usd": (
                len(cases) * STAGE5N_COST_PER_CASE_USD
            ),
            "configured_cost_guard_usd": DEFAULT_HOLDOUT_COST_GUARD_USD,
            "requires_explicit_execute_flag": True,
        }
    )
    if plan["max_provider_calls"] != MAX_HOLDOUT_PROVIDER_CALLS:
        raise ValueError("Stage 5O provider-call plan changed")
    return plan


def run_holdout(
    *,
    cases: list[MediQCase],
    spec: Mapping[str, Any],
    model: str,
    index_path: Path,
    provider_checkpoint: Path,
    output_path: Path,
    cost_guard_usd: float = DEFAULT_HOLDOUT_COST_GUARD_USD,
    reuse_only: bool = False,
) -> dict[str, Any]:
    fingerprint = holdout_fingerprint(cases, spec)
    provider = CheckpointedGeminiProvider(
        model=model,
        checkpoint_path=provider_checkpoint,
        fingerprint=fingerprint,
        max_calls=MAX_HOLDOUT_PROVIDER_CALLS,
        max_cost_usd=cost_guard_usd,
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_new_calls=not reuse_only,
    )
    retriever = TextbooksBM25Retriever(index_path)
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for position, case in enumerate(cases, start=1):
        result = _case_run(
            case,
            provider=provider,
            retriever=retriever,
            max_questions=int(spec["max_questions"]),
            top_k=int(spec["retrieval"]["top_k"]),
            diagnostic_conditions=HOLDOUT_DIAGNOSTIC_CONDITIONS,
            patient_factory=ConceptAwareFactPatient,
            question_refiner=CoverageAwareQuestionRefiner(),
        )
        results.append(result)
        report = {
            "format_version": 1,
            "stage": "5O",
            "benchmark": "MediQ handoff holdout",
            "dataset_fingerprint": fingerprint,
            "source_revision": spec["revision"],
            "source_sha256": spec["source_sha256"],
            "model": model,
            "cost_guard_usd": cost_guard_usd,
            "results": results,
            "metrics": summarize(
                results,
                provider,
                diagnostic_conditions=HOLDOUT_DIAGNOSTIC_CONDITIONS,
            ),
        }
        _save_json(output_path, report)
        print(
            f"[{position}/{len(cases)}] {case.case_id}: {result['status']} "
            f"questions={result['questions_asked']} cost=${provider.spent:.4f}"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_HOLDOUT_SPEC)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--provider-checkpoint", type=Path, default=DEFAULT_HOLDOUT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_HOLDOUT_OUTPUT)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--cost-guard", type=float, default=DEFAULT_HOLDOUT_COST_GUARD_USD)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Permit guarded provider calls after the dry-run plan is approved",
    )
    args = parser.parse_args(argv)

    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    spec = load_holdout_spec(args.spec)
    if args.download:
        download_source(args.source, spec)
    if not args.source.exists():
        raise FileNotFoundError("Pinned MediQ source is missing; rerun with --download")
    cases = load_holdout_cases(args.source, spec)
    plan = holdout_dry_run_plan(cases, spec, args.model)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.execute and not args.reuse_only:
        raise RuntimeError("Paid holdout is locked; review --dry-run then pass --execute")
    if plan["theoretical_cost_bound_usd"] > args.cost_guard:
        raise RuntimeError("Dry-run theoretical cost exceeds the configured guard")
    report = run_holdout(
        cases=cases,
        spec=spec,
        model=args.model,
        index_path=args.index,
        provider_checkpoint=args.provider_checkpoint,
        output_path=args.output,
        cost_guard_usd=args.cost_guard,
        reuse_only=args.reuse_only,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
