"""Structured answerability routing for the Agent v3 research path.

This module deliberately separates three responsibilities:

1. query completeness: are the patient facts required by the decision present?
2. evidence applicability: can the retrieved evidence support this patient/task?
3. routing: deterministically choose answer, clarify, abstain, or escalate.

The gate implementations are injected.  They may later be backed by a model or
an independently reviewed decision contract, but the router never delegates its
action policy to free-form model text.  Agent v2 remains unchanged.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph


ANSWERABILITY_SCHEMA_VERSION = 1
RISK_LEVELS = {"low", "medium", "high"}
ACTIONS = {"answer", "clarify", "abstain", "escalate"}
QUERY_ISSUES = {
    "none",
    "missing_required_facts",
    "ambiguous",
    "false_premise",
    "nonfactual",
    "unsupported_task",
}


QUERY_COMPLETENESS_PROMPT = """Classify whether the user supplied the patient facts required for the requested clinical decision.

QUESTION:
{query}

Return one JSON object with exactly these fields:
{{
  "schema_version": 1,
  "decision_requested": "short description",
  "known_patient_facts": ["required fact name"],
  "required_patient_facts": ["required fact name"],
  "missing_required_facts": ["required fact name"],
  "query_complete": true,
  "query_issue": "none|missing_required_facts|ambiguous|false_premise|nonfactual|unsupported_task",
  "risk_level": "low|medium|high",
  "clarification_questions": [],
  "reason": "brief reason"
}}

Rules:
- Judge query answerability only; do not judge retrieved evidence.
- known_patient_facts and missing_required_facts must partition required_patient_facts.
- query_complete is true only when query_issue is none and no required facts are missing.
- Use clarify only for missing facts or resolvable ambiguity; do not guess missing information.
"""


EVIDENCE_APPLICABILITY_PROMPT = """Classify whether the retrieved evidence can support an answer to this patient-specific question.

QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

Return one JSON object with exactly these fields:
{{
  "schema_version": 1,
  "evidence_available": true,
  "sufficient": true,
  "patient_applicable": true,
  "conflicting": false,
  "outdated": false,
  "tool_failed": false,
  "cited_evidence_ids": ["source id"],
  "reason": "brief reason"
}}

Rules:
- Relevance alone is not sufficiency or patient applicability.
- Mark conflicting when trusted passages support incompatible conclusions.
- Mark outdated when the evidence cannot support the time-sensitive decision.
- Do not answer the medical question; only classify the evidence.
"""


class InvalidGateDecision(ValueError):
    """Raised when a gate output violates the structured contract."""


def _string_list(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    # Tuples are accepted for internally persisted dataclass state; provider JSON
    # still arrives as a list.
    if not isinstance(value, (list, tuple)):
        raise InvalidGateDecision(f"{field} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidGateDecision(f"{field} must contain non-empty strings")
        text = re.sub(r"\s+", " ", item).strip()
        if text not in normalized:
            normalized.append(text)
    if not allow_empty and not normalized:
        raise InvalidGateDecision(f"{field} must not be empty")
    return tuple(normalized)


def _boolean(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise InvalidGateDecision(f"{field} must be a boolean")
    return value


def _text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidGateDecision(f"{field} must be a non-empty string")
    return re.sub(r"\s+", " ", value).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise InvalidGateDecision("Gate output must be a mapping or JSON object")
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidGateDecision(f"Gate output is not valid JSON: {error}") from error
    if not isinstance(parsed, Mapping):
        raise InvalidGateDecision("Gate JSON must be an object")
    return parsed


@dataclass(frozen=True)
class QueryCompletenessDecision:
    schema_version: int
    decision_requested: str
    known_patient_facts: tuple[str, ...]
    required_patient_facts: tuple[str, ...]
    missing_required_facts: tuple[str, ...]
    query_complete: bool
    query_issue: str
    risk_level: str
    clarification_questions: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class EvidenceApplicabilityDecision:
    schema_version: int
    evidence_available: bool
    sufficient: bool
    patient_applicable: bool
    conflicting: bool
    outdated: bool
    tool_failed: bool
    cited_evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RetrievalEvidence:
    contexts: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    latency_s: float = 0.0
    tool_failed: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError("retrieval latency cannot be negative")
        if len(self.evidence_ids) not in {0, len(self.contexts)}:
            raise ValueError("evidence_ids must be empty or align with contexts")
        if any(not item.strip() for item in self.contexts):
            raise ValueError("retrieval contexts must be non-empty strings")
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("evidence IDs must be non-empty strings")


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    reason: str
    response: str

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unsupported answerability action: {self.action}")


def parse_query_completeness(value: Any) -> QueryCompletenessDecision:
    if isinstance(value, QueryCompletenessDecision):
        value = asdict(value)
    payload = _mapping(value)
    if payload.get("schema_version") != ANSWERABILITY_SCHEMA_VERSION:
        raise InvalidGateDecision("Unsupported query-completeness schema version")

    known = _string_list(payload.get("known_patient_facts"), "known_patient_facts")
    required = _string_list(
        payload.get("required_patient_facts"), "required_patient_facts"
    )
    missing = _string_list(
        payload.get("missing_required_facts"), "missing_required_facts"
    )
    questions = _string_list(
        payload.get("clarification_questions"), "clarification_questions"
    )
    complete = _boolean(payload, "query_complete")
    query_issue = _text(payload, "query_issue").lower()
    if query_issue not in QUERY_ISSUES:
        raise InvalidGateDecision(f"Unsupported query_issue: {query_issue}")
    risk_level = _text(payload, "risk_level").lower()
    if risk_level not in RISK_LEVELS:
        raise InvalidGateDecision(f"Unsupported risk_level: {risk_level}")

    required_set = set(required)
    known_set = set(known)
    missing_set = set(missing)
    if known_set & missing_set:
        raise InvalidGateDecision("Known and missing facts must be disjoint")
    if known_set | missing_set != required_set:
        raise InvalidGateDecision(
            "Known and missing facts must partition required patient facts"
        )
    expected_complete = query_issue == "none" and not missing
    if complete != expected_complete:
        raise InvalidGateDecision("query_complete conflicts with query issue or missing facts")
    if bool(missing) != (query_issue == "missing_required_facts"):
        raise InvalidGateDecision(
            "missing facts require the missing_required_facts query issue"
        )
    if query_issue in {"missing_required_facts", "ambiguous"} and not questions:
        raise InvalidGateDecision("Missing facts require a clarification question")
    if query_issue not in {"missing_required_facts", "ambiguous"} and questions:
        raise InvalidGateDecision(
            "Only missing or ambiguous queries can request clarification"
        )

    return QueryCompletenessDecision(
        schema_version=ANSWERABILITY_SCHEMA_VERSION,
        decision_requested=_text(payload, "decision_requested"),
        known_patient_facts=known,
        required_patient_facts=required,
        missing_required_facts=missing,
        query_complete=complete,
        query_issue=query_issue,
        risk_level=risk_level,
        clarification_questions=questions,
        reason=_text(payload, "reason"),
    )


def parse_evidence_applicability(value: Any) -> EvidenceApplicabilityDecision:
    if isinstance(value, EvidenceApplicabilityDecision):
        value = asdict(value)
    payload = _mapping(value)
    if payload.get("schema_version") != ANSWERABILITY_SCHEMA_VERSION:
        raise InvalidGateDecision("Unsupported evidence-applicability schema version")

    available = _boolean(payload, "evidence_available")
    sufficient = _boolean(payload, "sufficient")
    applicable = _boolean(payload, "patient_applicable")
    citations = _string_list(
        payload.get("cited_evidence_ids"), "cited_evidence_ids"
    )
    if not available and (sufficient or applicable or citations):
        raise InvalidGateDecision(
            "Unavailable evidence cannot be sufficient, applicable, or cited"
        )
    if sufficient and not citations:
        raise InvalidGateDecision("Sufficient evidence requires at least one cited ID")

    return EvidenceApplicabilityDecision(
        schema_version=ANSWERABILITY_SCHEMA_VERSION,
        evidence_available=available,
        sufficient=sufficient,
        patient_applicable=applicable,
        conflicting=_boolean(payload, "conflicting"),
        outdated=_boolean(payload, "outdated"),
        tool_failed=_boolean(payload, "tool_failed"),
        cited_evidence_ids=citations,
        reason=_text(payload, "reason"),
    )


def route_answerability(
    query: QueryCompletenessDecision,
    evidence: EvidenceApplicabilityDecision | None = None,
) -> RoutingDecision:
    """Apply the Agent v3 action policy without another model decision."""
    if not query.query_complete:
        if query.query_issue in {"missing_required_facts", "ambiguous"}:
            questions = " ".join(query.clarification_questions)
            return RoutingDecision(
                action="clarify",
                reason="Required information is missing or the query is ambiguous.",
                response=(
                    f"I need more information before I can assess "
                    f"{query.decision_requested}. {questions}"
                ),
            )
        return RoutingDecision(
            action="abstain",
            reason=f"The query is not answerable: {query.query_issue}.",
            response=(
                "I cannot answer this request reliably because it contains a false "
                "premise, asks for a non-factual judgment, or falls outside the "
                "validated task."
            ),
        )
    if evidence is None:
        return RoutingDecision(
            action="abstain",
            reason="Evidence applicability was not established.",
            response=(
                "I cannot answer safely because the available evidence was not "
                "validated for this question."
            ),
        )
    if evidence.conflicting or evidence.outdated:
        issue = "conflicting" if evidence.conflicting else "outdated"
        return RoutingDecision(
            action="escalate",
            reason=f"The retrieved evidence is {issue}.",
            response=(
                f"The available medical evidence is {issue}, so this decision "
                "requires review by a qualified clinician using current sources."
            ),
        )
    if (
        evidence.tool_failed
        or not evidence.evidence_available
        or not evidence.sufficient
        or not evidence.patient_applicable
    ):
        return RoutingDecision(
            action="abstain",
            reason="Retrieved evidence is unavailable, insufficient, or not applicable.",
            response=(
                "I cannot answer safely because the retrieved evidence is unavailable, "
                "insufficient, or not applicable to this patient."
            ),
        )
    return RoutingDecision(
        action="answer",
        reason="The query is complete and the evidence is sufficient and applicable.",
        response="",
    )


class AnswerabilityAgentState(TypedDict):
    query: str
    retrieval_query: NotRequired[str]
    query_gate: NotRequired[dict[str, Any]]
    evidence_gate: NotRequired[dict[str, Any]]
    retrieved_context: list[str]
    retrieved_ids: list[str]
    retrieval_latency_s: float
    action: str
    final_answer: str
    status: str
    gate_errors: list[str]
    query_gate_calls: int
    evidence_gate_calls: int
    generation_calls: int
    _retrieval_evidence: NotRequired[dict[str, Any]]


QueryGate = Callable[[str], QueryCompletenessDecision | Mapping[str, Any] | str]
Retriever = Callable[[str], RetrievalEvidence]
EvidenceGate = Callable[
    [str, RetrievalEvidence], EvidenceApplicabilityDecision | Mapping[str, Any] | str
]
Generator = Callable[[str, RetrievalEvidence], str]


def _query_gate_node(state: AnswerabilityAgentState, gate: QueryGate) -> dict[str, Any]:
    try:
        decision = parse_query_completeness(gate(state["query"]))
        route = route_answerability(decision) if not decision.query_complete else None
        return {
            "query_gate": asdict(decision),
            "query_gate_calls": state.get("query_gate_calls", 0) + 1,
            "action": route.action if route else "",
            "final_answer": route.response if route else "",
            "status": route.action if route else "query_complete",
        }
    except Exception as error:
        message = f"query_completeness_error:{type(error).__name__}:{error}"
        return {
            "query_gate_calls": state.get("query_gate_calls", 0) + 1,
            "action": "abstain",
            "final_answer": (
                "I cannot answer safely because the required patient information "
                "could not be validated."
            ),
            "status": "query_gate_error",
            "gate_errors": state.get("gate_errors", []) + [message],
        }


def _route_after_query(state: AnswerabilityAgentState) -> str:
    return END if state.get("action") else "retrieve"


def _retrieve_node(
    state: AnswerabilityAgentState, retriever: Retriever
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        evidence = retriever(state.get("retrieval_query") or state["query"])
        if not isinstance(evidence, RetrievalEvidence):
            raise TypeError("Retriever must return RetrievalEvidence")
    except Exception as error:
        evidence = RetrievalEvidence(
            contexts=(),
            evidence_ids=(),
            latency_s=time.perf_counter() - started,
            tool_failed=True,
            error=f"{type(error).__name__}: {error}",
        )
    return {
        "retrieved_context": list(evidence.contexts),
        "retrieved_ids": list(evidence.evidence_ids),
        "retrieval_latency_s": evidence.latency_s,
        "status": "retrieval_failed" if evidence.tool_failed else "retrieved",
        "_retrieval_evidence": asdict(evidence),
    }


def _retrieval_from_state(state: AnswerabilityAgentState) -> RetrievalEvidence:
    raw = state.get("_retrieval_evidence")
    if isinstance(raw, Mapping):
        return RetrievalEvidence(
            contexts=tuple(raw.get("contexts", [])),
            evidence_ids=tuple(raw.get("evidence_ids", [])),
            latency_s=float(raw.get("latency_s", 0.0)),
            tool_failed=bool(raw.get("tool_failed", False)),
            error=str(raw.get("error", "")),
        )
    return RetrievalEvidence(
        contexts=tuple(state.get("retrieved_context", [])),
        evidence_ids=tuple(state.get("retrieved_ids", [])),
        latency_s=float(state.get("retrieval_latency_s", 0.0)),
        tool_failed=state.get("status") == "retrieval_failed",
    )


def _evidence_gate_node(
    state: AnswerabilityAgentState, gate: EvidenceGate
) -> dict[str, Any]:
    evidence = _retrieval_from_state(state)
    if evidence.tool_failed and not evidence.contexts:
        decision = EvidenceApplicabilityDecision(
            schema_version=ANSWERABILITY_SCHEMA_VERSION,
            evidence_available=False,
            sufficient=False,
            patient_applicable=False,
            conflicting=False,
            outdated=False,
            tool_failed=True,
            cited_evidence_ids=(),
            reason="The retrieval tool failed before returning evidence.",
        )
        route = route_answerability(
            parse_query_completeness(state["query_gate"]), decision
        )
        return {
            "evidence_gate": asdict(decision),
            "action": route.action,
            "final_answer": route.response,
            "status": route.action,
        }
    try:
        decision = parse_evidence_applicability(gate(state["query"], evidence))
        unknown_citations = set(decision.cited_evidence_ids) - set(evidence.evidence_ids)
        if unknown_citations:
            raise InvalidGateDecision(
                "Evidence gate cited IDs that were not retrieved: "
                + ", ".join(sorted(unknown_citations))
            )
        route = route_answerability(
            parse_query_completeness(state["query_gate"]), decision
        )
        return {
            "evidence_gate": asdict(decision),
            "evidence_gate_calls": state.get("evidence_gate_calls", 0) + 1,
            "action": route.action,
            "final_answer": route.response,
            "status": route.action if route.action != "answer" else "evidence_approved",
        }
    except Exception as error:
        message = f"evidence_applicability_error:{type(error).__name__}:{error}"
        return {
            "evidence_gate_calls": state.get("evidence_gate_calls", 0) + 1,
            "action": "abstain",
            "final_answer": (
                "I cannot answer safely because evidence applicability could not "
                "be validated."
            ),
            "status": "evidence_gate_error",
            "gate_errors": state.get("gate_errors", []) + [message],
        }


def _route_after_evidence(state: AnswerabilityAgentState) -> str:
    return "generate" if state.get("action") == "answer" else END


def _generate_node(
    state: AnswerabilityAgentState, generator: Generator
) -> dict[str, Any]:
    evidence = _retrieval_from_state(state)
    try:
        answer = generator(state["query"], evidence).strip()
        if not answer:
            raise ValueError("Generator returned an empty answer")
        return {
            "final_answer": answer,
            "status": "answered",
            "generation_calls": state.get("generation_calls", 0) + 1,
        }
    except Exception as error:
        message = f"generation_error:{type(error).__name__}:{error}"
        return {
            "action": "abstain",
            "final_answer": (
                "I cannot answer safely because generation failed after evidence "
                "validation."
            ),
            "status": "generation_error",
            "generation_calls": state.get("generation_calls", 0) + 1,
            "gate_errors": state.get("gate_errors", []) + [message],
        }


def build_answerability_agent(
    *,
    query_gate: QueryGate,
    retriever: Retriever,
    evidence_gate: EvidenceGate,
    generator: Generator,
):
    """Build the isolated Agent v3 graph with injected, testable dependencies."""
    graph = StateGraph(AnswerabilityAgentState)
    graph.add_node("query_gate", lambda state: _query_gate_node(state, query_gate))
    graph.add_node("retrieve", lambda state: _retrieve_node(state, retriever))
    graph.add_node(
        "evidence_gate", lambda state: _evidence_gate_node(state, evidence_gate)
    )
    graph.add_node("generate", lambda state: _generate_node(state, generator))
    graph.add_edge(START, "query_gate")
    graph.add_conditional_edges(
        "query_gate", _route_after_query, {"retrieve": "retrieve", END: END}
    )
    graph.add_edge("retrieve", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate", _route_after_evidence, {"generate": "generate", END: END}
    )
    graph.add_edge("generate", END)
    return graph.compile()


def initial_answerability_state(
    query: str, *, retrieval_query: str | None = None
) -> AnswerabilityAgentState:
    """Return an explicit initial state for traces and reproducible tests."""
    state: AnswerabilityAgentState = {
        "query": query,
        "retrieved_context": [],
        "retrieved_ids": [],
        "retrieval_latency_s": 0.0,
        "action": "",
        "final_answer": "",
        "status": "running",
        "gate_errors": [],
        "query_gate_calls": 0,
        "evidence_gate_calls": 0,
        "generation_calls": 0,
    }
    if retrieval_query is not None:
        state["retrieval_query"] = retrieval_query
    return state
