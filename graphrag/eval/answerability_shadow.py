"""Run a zero-call shadow evaluation of a hand-crafted answerability gate.

This diagnostic intentionally operates on the already-observed Stage 5I/5J
development cases.  It does not call a model, retrieve new evidence, or claim
holdout generalization.  Its purpose is to test whether an explicit query-
completeness contract would have intercepted the known abstention failures.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from graphrag.eval.robustness_benchmark import (
    ABSTENTION_NEGATIVE_TYPES,
    DEFAULT_DATASET,
    DEFAULT_STAGE5J_JUDGMENTS,
    DEFAULT_STAGE5J_RESULTS,
    RobustnessCase,
    fingerprint,
    load_pilot_cases,
)


ANSWERABILITY_SCHEMA_VERSION = 1
DEFAULT_DECISION_SPEC = Path(__file__).parent / "specs" / "answerability_decisions.yaml"
DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "external"
    / "robustness"
    / "stage5k_answerability_shadow.json"
)
QUERY_COMPLETENESS_TYPES = ABSTENTION_NEGATIVE_TYPES | {"abstention"}
POSITIVE_ACTIONS = {"clarify", "abstain"}


@dataclass(frozen=True)
class RequiredFact:
    name: str
    description: str
    evidence_patterns: tuple[str, ...]
    clarification: str


@dataclass(frozen=True)
class DecisionContract:
    name: str
    decision_requested: str
    risk_level: str
    trigger_patterns: tuple[str, ...]
    required_patient_facts: tuple[RequiredFact, ...]


@dataclass(frozen=True)
class GateDecision:
    case_id: str
    decision_type: str
    decision_requested: str
    known_patient_facts: list[str]
    required_patient_facts: list[str]
    missing_required_facts: list[str]
    evidence_applicable: bool | None
    risk_level: str
    action: str
    reason: str
    clarification: str


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be a non-empty string list")
    return tuple(item.strip() for item in value)


def load_decision_contracts(
    path: Path = DEFAULT_DECISION_SPEC,
) -> tuple[dict[str, Any], list[DecisionContract]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Answerability specification must be a mapping")
    if payload.get("schema_version") != ANSWERABILITY_SCHEMA_VERSION:
        raise ValueError("Unsupported answerability schema version")
    if payload.get("scope") != "diagnostic_only":
        raise ValueError("Answerability specification must declare diagnostic_only scope")
    raw_contracts = payload.get("decision_types")
    if not isinstance(raw_contracts, Mapping) or not raw_contracts:
        raise ValueError("Answerability specification requires decision_types")

    contracts: list[DecisionContract] = []
    for name, raw_contract in raw_contracts.items():
        if not isinstance(raw_contract, Mapping):
            raise ValueError(f"Decision contract {name} must be a mapping")
        raw_facts = raw_contract.get("required_patient_facts")
        if not isinstance(raw_facts, Mapping) or not raw_facts:
            raise ValueError(f"Decision contract {name} requires patient facts")
        facts: list[RequiredFact] = []
        for fact_name, raw_fact in raw_facts.items():
            if not isinstance(raw_fact, Mapping):
                raise ValueError(f"Required fact {name}.{fact_name} must be a mapping")
            facts.append(
                RequiredFact(
                    name=str(fact_name),
                    description=str(raw_fact.get("description", "")).strip(),
                    evidence_patterns=_string_list(
                        raw_fact.get("evidence_patterns"),
                        f"{name}.{fact_name}.evidence_patterns",
                    ),
                    clarification=str(raw_fact.get("clarification", "")).strip(),
                )
            )
        contract = DecisionContract(
            name=str(name),
            decision_requested=str(raw_contract.get("decision_requested", "")).strip(),
            risk_level=str(raw_contract.get("risk_level", "")).strip(),
            trigger_patterns=_string_list(
                raw_contract.get("trigger_patterns"), f"{name}.trigger_patterns"
            ),
            required_patient_facts=tuple(facts),
        )
        if not contract.decision_requested or contract.risk_level not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(f"Decision contract {name} has invalid metadata")
        for pattern in contract.trigger_patterns:
            re.compile(pattern, re.IGNORECASE)
        for fact in contract.required_patient_facts:
            if not fact.description or not fact.clarification:
                raise ValueError(f"Required fact {name}.{fact.name} is incomplete")
            for pattern in fact.evidence_patterns:
                re.compile(pattern, re.IGNORECASE)
        contracts.append(contract)
    return dict(payload), contracts


def _matches_all(text: str, patterns: tuple[str, ...]) -> bool:
    return all(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _clarification(decision: str, missing: list[RequiredFact]) -> str:
    requested = "; ".join(fact.clarification for fact in missing)
    return (
        f"I cannot safely determine {decision} because required information is "
        f"missing. Please provide {requested}."
    )


def evaluate_case(
    case: RobustnessCase,
    contracts: list[DecisionContract],
    *,
    retrieval_available: bool = True,
) -> GateDecision:
    matches = [
        contract
        for contract in contracts
        if _matches_all(case.question, contract.trigger_patterns)
    ]
    if len(matches) > 1:
        names = ", ".join(contract.name for contract in matches)
        raise ValueError(f"Ambiguous answerability contracts for {case.case_id}: {names}")
    if not matches:
        return GateDecision(
            case_id=case.case_id,
            decision_type="unsupported",
            decision_requested="unsupported clinical decision",
            known_patient_facts=[],
            required_patient_facts=[],
            missing_required_facts=[],
            evidence_applicable=None,
            risk_level="high",
            action="abstain",
            reason="No validated decision contract covers this question.",
            clarification=(
                "I cannot safely answer because this prototype has no validated "
                "answerability contract for the requested decision."
            ),
        )

    contract = matches[0]
    present: list[RequiredFact] = []
    missing: list[RequiredFact] = []
    for fact in contract.required_patient_facts:
        if any(
            re.search(pattern, case.question, re.IGNORECASE)
            for pattern in fact.evidence_patterns
        ):
            present.append(fact)
        else:
            missing.append(fact)

    if not retrieval_available:
        action = "abstain"
        reason = "Required retrieval evidence is unavailable."
        clarification = (
            "I cannot answer safely because medical evidence retrieval is unavailable."
        )
        evidence_applicable: bool | None = False
    elif missing:
        action = "clarify"
        reason = "The query omits required patient-specific information."
        clarification = _clarification(contract.decision_requested, missing)
        evidence_applicable = None
    else:
        action = "answer"
        reason = "All required query fields for this diagnostic contract are present."
        clarification = ""
        # Applicability requires inspection of retrieved evidence, which this
        # zero-call query-completeness gate intentionally does not perform.
        evidence_applicable = None

    return GateDecision(
        case_id=case.case_id,
        decision_type=contract.name,
        decision_requested=contract.decision_requested,
        known_patient_facts=[fact.name for fact in present],
        required_patient_facts=[
            fact.name for fact in contract.required_patient_facts
        ],
        missing_required_facts=[fact.name for fact in missing],
        evidence_applicable=evidence_applicable,
        risk_level=contract.risk_level,
        action=action,
        reason=reason,
        clarification=clarification,
    )


def _binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Binary metrics require equally sized, non-empty inputs")
    true_positive = sum(gold and guess for gold, guess in zip(expected, predicted))
    false_positive = sum(not gold and guess for gold, guess in zip(expected, predicted))
    false_negative = sum(gold and not guess for gold, guess in zip(expected, predicted))
    true_negative = sum(
        not gold and not guess for gold, guess in zip(expected, predicted)
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "n": len(expected),
        "n_positive": sum(expected),
        "n_negative": sum(not value for value in expected),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_abstention_rate": (
            false_positive / sum(not value for value in expected)
            if any(not value for value in expected)
            else None
        ),
    }


def _load_saved_agent_abstentions(
    path: Path,
    dataset_fingerprint: str,
) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("generation_dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Saved judgment dataset fingerprint mismatch")
    abstentions: dict[str, bool] = {}
    for judgment in payload.get("judgments", []):
        systems = judgment.get("systems", {})
        agent = systems.get("matched-agent", {})
        if "abstained" in agent:
            abstentions[str(judgment["case_id"])] = bool(agent["abstained"])
    return abstentions


def _load_saved_agent_retrieval_status(
    path: Path, dataset_fingerprint: str
) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Saved generation dataset fingerprint mismatch")
    systems = set(payload.get("systems", []))
    if "matched-agent" not in systems:
        raise ValueError("Saved generation report lacks matched-agent outputs")
    status: dict[str, bool] = {}
    for result in payload.get("results", []):
        if result.get("system") != "matched-agent":
            continue
        status[str(result["case_id"])] = not bool(result.get("tool_errors"))
    return status


def run_shadow_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    generation_path: Path = DEFAULT_STAGE5J_RESULTS,
    judgments_path: Path = DEFAULT_STAGE5J_JUDGMENTS,
    decision_spec_path: Path = DEFAULT_DECISION_SPEC,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    dataset, cases = load_pilot_cases(dataset_path)
    dataset_fingerprint = str(dataset["metadata"]["dataset_fingerprint"])
    retrieval_available = _load_saved_agent_retrieval_status(
        generation_path, dataset_fingerprint
    )
    spec, contracts = load_decision_contracts(decision_spec_path)
    saved_abstentions = _load_saved_agent_abstentions(
        judgments_path, dataset_fingerprint
    )

    missing_generation = [
        case.case_id for case in cases if case.case_id not in retrieval_available
    ]
    if missing_generation:
        raise ValueError(f"Saved generation report is incomplete: {missing_generation}")
    decisions = [
        evaluate_case(
            case,
            contracts,
            retrieval_available=retrieval_available[case.case_id],
        )
        for case in cases
    ]
    by_id = {decision.case_id: decision for decision in decisions}
    labelable = [case for case in cases if case.test_type in QUERY_COMPLETENESS_TYPES]
    missing_judgments = [
        case.case_id for case in labelable if case.case_id not in saved_abstentions
    ]
    if missing_judgments:
        raise ValueError(f"Saved judgments are incomplete: {missing_judgments}")

    expected = [case.test_type == "abstention" for case in labelable]
    gate_predicted = [by_id[case.case_id].action in POSITIVE_ACTIONS for case in labelable]
    agent_predicted = [saved_abstentions[case.case_id] for case in labelable]
    tool_cases = [case for case in cases if case.test_type == "vector_tool_failure"]
    unsupported = [
        decision.case_id for decision in decisions if decision.decision_type == "unsupported"
    ]

    report = {
        "evaluation_type": "offline_answerability_shadow",
        "schema_version": ANSWERABILITY_SCHEMA_VERSION,
        "scope": "development_diagnostic_not_holdout",
        "external_model_calls": 0,
        "estimated_api_cost_usd": 0.0,
        "dataset_fingerprint": dataset_fingerprint,
        "generation_checkpoint": str(generation_path),
        "judgment_checkpoint": str(judgments_path),
        "decision_spec": str(decision_spec_path),
        "decision_spec_fingerprint": fingerprint(spec),
        "case_count": len(cases),
        "query_completeness_case_count": len(labelable),
        "decision_contract_count": len(contracts),
        "unsupported_case_count": len(unsupported),
        "unsupported_case_ids": unsupported,
        "metrics": {
            "saved_agent": _binary_metrics(expected, agent_predicted),
            "shadow_gate": _binary_metrics(expected, gate_predicted),
            "tool_failure_safe_action": {
                "n": len(tool_cases),
                "passed": sum(
                    by_id[case.case_id].action == "abstain" for case in tool_cases
                ),
                "rate": (
                    sum(by_id[case.case_id].action == "abstain" for case in tool_cases)
                    / len(tool_cases)
                    if tool_cases
                    else None
                ),
            },
        },
        "decisions": [asdict(decision) for decision in decisions],
        "limitations": [
            "The five decision contracts were written after inspecting these development cases.",
            "The four abstention labels and clinical required fields are not clinician reviewed.",
            "Evidence applicability is not evaluated by this query-only offline gate.",
            "Saved Agent abstention labels come from an uncalibrated LLM judge with deterministic guards.",
        ],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--generation-report", type=Path, default=DEFAULT_STAGE5J_RESULTS)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_STAGE5J_JUDGMENTS)
    parser.add_argument("--decision-spec", type=Path, default=DEFAULT_DECISION_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_shadow_evaluation(
        dataset_path=args.dataset,
        generation_path=args.generation_report,
        judgments_path=args.judgments,
        decision_spec_path=args.decision_spec,
        output_path=args.output,
    )
    saved = report["metrics"]["saved_agent"]
    gate = report["metrics"]["shadow_gate"]
    tool = report["metrics"]["tool_failure_safe_action"]
    print("Stage 5K offline answerability shadow evaluation")
    print(f"Cases: {report['case_count']} (query completeness: {gate['n']})")
    print(
        "Saved Agent abstention precision/recall/F1: "
        f"{saved['precision']} / {saved['recall']:.3f} / {saved['f1']:.3f}"
    )
    print(
        "Shadow gate abstention precision/recall/F1: "
        f"{gate['precision']:.3f} / {gate['recall']:.3f} / {gate['f1']:.3f}"
    )
    print(
        "Shadow false-abstention rate: "
        f"{gate['false_abstention_rate']:.3f} ({gate['false_positive']}/{gate['n_negative']})"
    )
    print(f"Tool-failure safe action: {tool['passed']}/{tool['n']}")
    print("External model calls: 0; estimated API cost: US$0.00")
    print(f"Diagnostic report: {args.output}")


if __name__ == "__main__":
    main()
