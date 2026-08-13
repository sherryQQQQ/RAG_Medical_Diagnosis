"""Bounded interactive clinical Agent with a provenance-aware handoff.

Flow:
    update handoff + plan one question -> ask patient (at most 3 times)
    -> retrieve evidence -> fresh diagnostician -> safety gate

All model and tool dependencies are injected.  The diagnostic dependency sees
only ``DiagnosticPacket``; it cannot inherit the interviewer model's hidden
reasoning or unstructured message history.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from graphrag.agent.clinical_handoff import (
    ClinicalHandoff,
    ConversationTurn,
    DiagnosticDraft,
    DiagnosticPacket,
    EvidenceItem,
    InterviewDecision,
    InterviewUpdate,
    SafetyDecision,
)
from graphrag.agent.clinical_tools import ClinicalTools


HARD_MAX_INTERVIEW_QUESTIONS = 3

Interviewer = Callable[
    [tuple[ConversationTurn, ...], ClinicalHandoff | None], InterviewUpdate
]
Diagnostician = Callable[[DiagnosticPacket], DiagnosticDraft]
SafetyValidator = Callable[[DiagnosticPacket, DiagnosticDraft], SafetyDecision]


class ClinicalAgentState(TypedDict):
    conversation: list[ConversationTurn]
    handoff: NotRequired[ClinicalHandoff]
    interview_decision: NotRequired[InterviewDecision]
    diagnostic_packet: NotRequired[DiagnosticPacket]
    diagnostic_draft: NotRequired[DiagnosticDraft]
    safety_decision: NotRequired[SafetyDecision]
    questions_asked: int
    max_questions: int
    force_diagnosis_on_budget: bool
    allow_incomplete_finalize: bool
    include_source_turns: bool
    pending_question: str
    action: str
    final_answer: str
    status: str
    errors: list[str]
    trace: list[str]
    patient_tool_calls: int
    patient_tool_successes: int
    retrieval_tool_calls: int
    retrieval_tool_successes: int
    interview_calls: int
    diagnostic_calls: int
    safety_calls: int


def _fail_closed(
    state: ClinicalAgentState, *, status: str, error: Exception | str
) -> dict[str, Any]:
    detail = str(error)
    return {
        "action": "abstain",
        "final_answer": (
            "I cannot provide a reliable assessment because the clinical "
            "handoff could not be completed safely."
        ),
        "status": status,
        "errors": state.get("errors", []) + [f"{status}:{detail}"],
        "trace": state.get("trace", []) + [status],
    }


def _validate_handoff_sources(
    handoff: ClinicalHandoff, conversation: tuple[ConversationTurn, ...]
) -> None:
    patient_ids = {turn.turn_id for turn in conversation if turn.role == "patient"}
    for fact in handoff.facts:
        unknown = set(fact.source_turn_ids) - patient_ids
        if unknown:
            raise ValueError(
                f"Fact {fact.fact_id} cites unknown patient turns: {sorted(unknown)}"
            )


def _interview_node(
    state: ClinicalAgentState, interviewer: Interviewer
) -> dict[str, Any]:
    try:
        conversation = tuple(state["conversation"])
        update = interviewer(conversation, state.get("handoff"))
        if not isinstance(update, InterviewUpdate):
            raise TypeError("Interviewer must return InterviewUpdate")
        handoff = update.handoff
        decision = update.decision
        _validate_handoff_sources(handoff, conversation)
        result: dict[str, Any] = {
            "handoff": handoff,
            "interview_decision": decision,
            "interview_calls": state.get("interview_calls", 0) + 1,
            "pending_question": decision.question,
            "status": (
                "question_planned" if decision.action == "ask" else "interview_complete"
            ),
            "trace": state.get("trace", []) + [f"interview_{decision.action}"],
        }
        if decision.action == "ask" and state["questions_asked"] >= state["max_questions"]:
            missing = ", ".join(handoff.missing_information) or "decisive patient facts"
            if state["force_diagnosis_on_budget"]:
                forced = InterviewDecision(
                    action="finalize",
                    reason=(
                        "The benchmark question budget was exhausted; continue with "
                        f"a best-effort forced choice while recording missing: {missing}."
                    ),
                )
                result.update(
                    {
                        "interview_decision": forced,
                        "pending_question": "",
                        "status": "question_budget_forced_diagnosis",
                        "trace": state.get("trace", [])
                        + ["interview_ask", "question_budget_forced_diagnosis"],
                    }
                )
            else:
                result.update(
                    {
                        "action": "abstain",
                        "final_answer": (
                            "I cannot provide a reliable assessment within the question "
                            f"budget because information is still missing: {missing}."
                        ),
                        "status": "question_budget_exhausted",
                        "trace": state.get("trace", [])
                        + ["interview_ask", "question_budget_exhausted"],
                    }
                )
        if (
            decision.action == "finalize"
            and not handoff.ready_for_diagnosis
            and not (
                state["force_diagnosis_on_budget"]
                and state["questions_asked"] >= state["max_questions"]
            )
            and not state["allow_incomplete_finalize"]
        ):
            result.update(
                {
                    "action": "abstain",
                    "final_answer": (
                        "I cannot provide a reliable assessment because decisive "
                        "patient information is still missing."
                    ),
                    "status": "incomplete_handoff",
                }
            )
        return result
    except Exception as error:
        update = _fail_closed(state, status="interview_error", error=error)
        update["interview_calls"] = state.get("interview_calls", 0) + 1
        return update


def _route_after_interview(state: ClinicalAgentState) -> str:
    if state.get("action"):
        return END
    decision = state.get("interview_decision")
    return "ask_patient" if decision and decision.action == "ask" else "retrieve"


def _ask_patient_node(
    state: ClinicalAgentState, tools: ClinicalTools
) -> dict[str, Any]:
    calls = state.get("patient_tool_calls", 0) + 1
    try:
        response = " ".join(
            tools.ask_patient(
                state["pending_question"], tuple(state["conversation"])
            ).split()
        )
        if not response:
            raise ValueError("Patient tool returned an empty response")
        question_number = state["questions_asked"] + 1
        conversation = state["conversation"] + [
            ConversationTurn(
                turn_id=f"agent-{question_number}",
                role="agent",
                content=state["pending_question"],
            ),
            ConversationTurn(
                turn_id=f"patient-{question_number}",
                role="patient",
                content=response,
            ),
        ]
        return {
            "conversation": conversation,
            "questions_asked": question_number,
            "pending_question": "",
            "patient_tool_calls": calls,
            "patient_tool_successes": state.get("patient_tool_successes", 0) + 1,
            "status": "patient_answered",
            "trace": state.get("trace", []) + ["patient_tool_success"],
        }
    except Exception as error:
        update = _fail_closed(state, status="patient_tool_error", error=error)
        update["patient_tool_calls"] = calls
        return update


def _route_after_patient_tool(state: ClinicalAgentState) -> str:
    return END if state.get("action") else "interview"


def _retrieve_node(
    state: ClinicalAgentState, tools: ClinicalTools
) -> dict[str, Any]:
    calls = state.get("retrieval_tool_calls", 0) + 1
    try:
        evidence = tools.retrieve_evidence(state["handoff"])
        if not isinstance(evidence, tuple) or not all(
            isinstance(item, EvidenceItem) for item in evidence
        ):
            raise TypeError("Retrieval tool must return tuple[EvidenceItem, ...]")
        if not evidence:
            raise ValueError("Retrieval tool returned no evidence")
        patient_source_ids = {
            source_id
            for fact in state["handoff"].facts
            for source_id in fact.source_turn_ids
        }
        source_turns = (
            tuple(
                turn
                for turn in state["conversation"]
                if turn.turn_id in patient_source_ids
            )
            if state["include_source_turns"]
            else ()
        )
        packet = DiagnosticPacket(
            handoff=state["handoff"], evidence=evidence, source_turns=source_turns
        )
        return {
            "diagnostic_packet": packet,
            "retrieval_tool_calls": calls,
            "retrieval_tool_successes": state.get("retrieval_tool_successes", 0) + 1,
            "status": "evidence_retrieved",
            "trace": state.get("trace", []) + ["retrieval_tool_success"],
        }
    except Exception as error:
        update = _fail_closed(state, status="retrieval_tool_error", error=error)
        update["retrieval_tool_calls"] = calls
        return update


def _route_after_retrieval(state: ClinicalAgentState) -> str:
    return END if state.get("action") else "diagnose"


def _diagnose_node(
    state: ClinicalAgentState, diagnostician: Diagnostician
) -> dict[str, Any]:
    try:
        packet = state["diagnostic_packet"]
        draft = diagnostician(packet)
        if not isinstance(draft, DiagnosticDraft):
            raise TypeError("Diagnostician must return DiagnosticDraft")
        known_facts = {fact.fact_id for fact in packet.handoff.facts}
        known_evidence = {item.evidence_id for item in packet.evidence}
        unknown_facts = set(draft.cited_fact_ids) - known_facts
        unknown_evidence = set(draft.cited_evidence_ids) - known_evidence
        if unknown_facts or unknown_evidence:
            raise ValueError(
                "Diagnostic draft contains unknown citations: "
                f"facts={sorted(unknown_facts)}, evidence={sorted(unknown_evidence)}"
            )
        if not draft.cited_fact_ids or not draft.cited_evidence_ids:
            raise ValueError("Diagnostic draft must cite patient facts and evidence")
        return {
            "diagnostic_draft": draft,
            "diagnostic_calls": state.get("diagnostic_calls", 0) + 1,
            "status": "diagnostic_draft_ready",
            "trace": state.get("trace", []) + ["diagnosed"],
        }
    except Exception as error:
        update = _fail_closed(state, status="diagnostic_error", error=error)
        update["diagnostic_calls"] = state.get("diagnostic_calls", 0) + 1
        return update


def _route_after_diagnosis(state: ClinicalAgentState) -> str:
    return END if state.get("action") else "safety_gate"


def _safety_node(
    state: ClinicalAgentState, validator: SafetyValidator
) -> dict[str, Any]:
    try:
        decision = validator(state["diagnostic_packet"], state["diagnostic_draft"])
        if not isinstance(decision, SafetyDecision):
            raise TypeError("Safety validator must return SafetyDecision")
        if decision.action == "approve":
            action = "answer"
            answer = state["diagnostic_draft"].answer
            status = "answered"
        elif decision.action == "escalate":
            action = "escalate"
            answer = (
                "The available information requires review by a qualified clinician "
                "before a recommendation can be made."
            )
            status = "escalated"
        else:
            action = "abstain"
            answer = (
                "I cannot provide a reliable assessment from the available "
                "patient information and evidence."
            )
            status = "abstained"
        return {
            "safety_decision": decision,
            "safety_calls": state.get("safety_calls", 0) + 1,
            "action": action,
            "final_answer": answer,
            "status": status,
            "trace": state.get("trace", []) + [f"safety_{decision.action}"],
        }
    except Exception as error:
        update = _fail_closed(state, status="safety_gate_error", error=error)
        update["safety_calls"] = state.get("safety_calls", 0) + 1
        return update


def build_clinical_handoff_agent(
    *,
    interview: Interviewer,
    tools: ClinicalTools,
    diagnose: Diagnostician,
    validate_safety: SafetyValidator,
):
    """Compile the dependency-injected LangGraph workflow."""
    graph = StateGraph(ClinicalAgentState)
    graph.add_node("interview", lambda state: _interview_node(state, interview))
    graph.add_node("ask_patient", lambda state: _ask_patient_node(state, tools))
    graph.add_node("retrieve", lambda state: _retrieve_node(state, tools))
    graph.add_node("diagnose", lambda state: _diagnose_node(state, diagnose))
    graph.add_node("safety_gate", lambda state: _safety_node(state, validate_safety))
    graph.add_edge(START, "interview")
    graph.add_conditional_edges(
        "interview",
        _route_after_interview,
        {"ask_patient": "ask_patient", "retrieve": "retrieve", END: END},
    )
    graph.add_conditional_edges(
        "ask_patient",
        _route_after_patient_tool,
        {"interview": "interview", END: END},
    )
    graph.add_conditional_edges(
        "retrieve", _route_after_retrieval, {"diagnose": "diagnose", END: END}
    )
    graph.add_conditional_edges(
        "diagnose", _route_after_diagnosis, {"safety_gate": "safety_gate", END: END}
    )
    graph.add_edge("safety_gate", END)
    return graph.compile()


def initial_clinical_state(
    patient_message: str,
    *,
    max_questions: int = HARD_MAX_INTERVIEW_QUESTIONS,
    include_source_turns: bool = True,
    force_diagnosis_on_budget: bool = False,
    allow_incomplete_finalize: bool = False,
) -> ClinicalAgentState:
    """Create explicit state for a new bounded interview."""
    if not 0 <= max_questions <= HARD_MAX_INTERVIEW_QUESTIONS:
        raise ValueError(
            f"max_questions must be between 0 and {HARD_MAX_INTERVIEW_QUESTIONS}"
        )
    initial_turn = ConversationTurn(
        turn_id="patient-0", role="patient", content=patient_message
    )
    return {
        "conversation": [initial_turn],
        "questions_asked": 0,
        "max_questions": max_questions,
        "force_diagnosis_on_budget": force_diagnosis_on_budget,
        "allow_incomplete_finalize": allow_incomplete_finalize,
        "include_source_turns": include_source_turns,
        "pending_question": "",
        "action": "",
        "final_answer": "",
        "status": "running",
        "errors": [],
        "trace": [],
        "patient_tool_calls": 0,
        "patient_tool_successes": 0,
        "retrieval_tool_calls": 0,
        "retrieval_tool_successes": 0,
        "interview_calls": 0,
        "diagnostic_calls": 0,
        "safety_calls": 0,
    }


def serializable_clinical_result(state: ClinicalAgentState) -> dict[str, Any]:
    """Return a JSON-safe trace without exposing model-internal reasoning."""
    output = dict(state)
    for key in (
        "conversation",
        "handoff",
        "interview_decision",
        "diagnostic_packet",
        "diagnostic_draft",
        "safety_decision",
    ):
        value = output.get(key)
        if value is not None:
            if isinstance(value, list):
                output[key] = [asdict(item) for item in value]
            else:
                output[key] = asdict(value)
    return output
