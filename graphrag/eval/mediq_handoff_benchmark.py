"""Low-cost MediQ pilot for the provenance-aware clinical handoff Agent.

The pilot uses a deterministic local patient-fact selector so the only paid
calls are the interviewer and three final diagnostic conditions.  It is an
engineering and hypothesis-generation experiment, not a clinical validation.
Every provider response is checkpointed before the next call.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

# Patient conversations and raw model outputs remain local even when the user's
# general project environment enables LangSmith tracing.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from graphrag.agent.clinical_agent import (
    build_clinical_handoff_agent,
    initial_clinical_state,
)
from graphrag.agent.clinical_handoff import (
    ClinicalFact,
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
from graphrag.eval.checkpointed_gemini import (
    CheckpointedGeminiProvider,
    Provider,
    ProviderResponse,
)
from graphrag.eval.mediq_handoff_data import (
    DEFAULT_ROOT,
    DEFAULT_SOURCE,
    DEFAULT_SPEC,
    DIAGNOSTIC_CONDITIONS,
    DeterministicFactPatient,
    MediQCase,
    dataset_fingerprint,
    download_source,
    load_cases,
    load_spec,
    source_url,
    token_f1,
)
from graphrag.eval.mirage_benchmark import (
    MODEL_PRICING_USD_PER_MILLION,
    _mcnemar_exact_p,
    _wilson_interval,
    parse_answer_choice,
)
from graphrag.eval.mirage_corpus import DEFAULT_INDEX, TextbooksBM25Retriever


DEFAULT_PROVIDER_CHECKPOINT = DEFAULT_ROOT / "stage5n_provider_checkpoint.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "stage5n_handoff_results.json"
MAX_PROVIDER_CALLS = 35
MAX_PROMPT_CHARS = 14_000
MAX_OUTPUT_TOKENS = 1_024
DEFAULT_COST_GUARD_USD = 0.25


INTERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "chief_complaint": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_id": {"type": "string"},
                    "category": {"type": "string"},
                    "statement": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["present", "absent", "unknown"],
                    },
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "fact_id",
                    "category",
                    "statement",
                    "status",
                    "source_turn_ids",
                ],
            },
        },
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "ready_for_diagnosis": {"type": "boolean"},
        "action": {"type": "string", "enum": ["ask", "finalize"]},
        "reason": {"type": "string"},
        "question": {"type": "string"},
        "target_information": {"type": "string"},
    },
    "required": [
        "chief_complaint",
        "facts",
        "missing_information",
        "contradictions",
        "ready_for_diagnosis",
        "action",
        "reason",
        "question",
        "target_information",
    ],
}


DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_choice": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "answer": {"type": "string"},
        "differential": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "cited_patient_ids": {"type": "array", "items": {"type": "string"}},
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "answer_choice",
        "answer",
        "differential",
        "recommendations",
        "cited_patient_ids",
        "cited_evidence_ids",
        "confidence",
    ],
}


def diagnosis_schema(options: Mapping[str, str]) -> dict[str, Any]:
    """Constrain answer choice to the choices actually supplied by this case."""
    schema = copy.deepcopy(DIAGNOSIS_SCHEMA)
    schema["properties"]["answer_choice"]["enum"] = sorted(options)
    return schema


def _json_object(raw: str) -> Mapping[str, Any]:
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("Provider output must be a JSON object")
    return value


def parse_interview_update(
    raw: str, conversation: tuple[ConversationTurn, ...]
) -> InterviewUpdate:
    payload = _json_object(raw)
    missing_information = tuple(
        str(value) for value in payload["missing_information"]
    )
    contradictions = tuple(str(value) for value in payload["contradictions"])
    # Provider schemas constrain types but do not enforce cross-field logic.
    # Normalize readiness downward; never erase reported missing information.
    ready_for_diagnosis = (
        bool(payload["ready_for_diagnosis"])
        and not missing_information
        and not contradictions
    )
    facts = tuple(
        ClinicalFact(
            fact_id=str(item["fact_id"]),
            category=str(item["category"]),
            statement=str(item["statement"]),
            status=str(item["status"]),
            source_turn_ids=tuple(str(value) for value in item["source_turn_ids"]),
        )
        for item in payload["facts"]
    )
    handoff = ClinicalHandoff(
        chief_complaint=str(payload["chief_complaint"]),
        facts=facts,
        missing_information=missing_information,
        contradictions=contradictions,
        ready_for_diagnosis=ready_for_diagnosis,
    )
    valid_patient_turns = {turn.turn_id for turn in conversation if turn.role == "patient"}
    for fact in handoff.facts:
        if not set(fact.source_turn_ids) <= valid_patient_turns:
            raise ValueError(f"Interview fact {fact.fact_id} cites an unknown patient turn")
    decision = InterviewDecision(
        action=str(payload["action"]),
        reason=str(payload["reason"]),
        question=str(payload["question"]),
        target_information=str(payload["target_information"]),
    )
    return InterviewUpdate(handoff=handoff, decision=decision)


def _format_options(options: Mapping[str, str]) -> str:
    return "\n".join(f"{key}. {value}" for key, value in options.items())


def _interview_prompt(
    case: MediQCase,
    conversation: tuple[ConversationTurn, ...],
    previous: ClinicalHandoff | None,
    max_questions: int,
) -> str:
    questions_used = sum(turn.role == "agent" for turn in conversation)
    transcript = json.dumps([asdict(turn) for turn in conversation], indent=2)
    previous_text = json.dumps(asdict(previous), indent=2) if previous else "null"
    return f"""You are the interviewing component of a medical multiple-choice research benchmark.
Update the structured patient handoff using ONLY explicit patient messages in the transcript, then either ask one targeted question or finalize. Do not diagnose, answer the multiple-choice question, or use hidden patient facts.

Benchmark question:
{case.question}

Answer options (for deciding what information is discriminative, not for answering):
{_format_options(case.options)}

Questions used: {questions_used}/{max_questions}
Patient transcript:
{transcript}

Previous handoff:
{previous_text}

Rules:
- Preserve useful prior facts, but correct them if the new patient message contradicts them.
- Every fact must cite one or more existing patient turn IDs; never cite agent turns.
- Use stable fact IDs when retaining a fact.
- status must be present, absent, or unknown.
- Ask exactly one concise question that targets the most decision-relevant missing information.
- If enough information is present, set action=finalize, question="", target_information="".
- If action=finalize and ready_for_diagnosis=true, missing_information and contradictions must be empty.
- At the question limit you may still identify missing information; the benchmark router will force a best-effort choice without pretending the handoff is complete.
Return only the required JSON object."""


def _evidence_text(evidence: tuple[EvidenceItem, ...]) -> str:
    blocks: list[str] = []
    used = 0
    for item in evidence:
        block = f"[{item.evidence_id}] {item.source}\n{item.text[:900]}"
        if used + len(block) > 7_500:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def _truncate_headtail(
    conversation: tuple[ConversationTurn, ...], budget_chars: int
) -> tuple[str, set[str]]:
    """Keep whole turns from the head and tail of the transcript within a
    character budget (Phase 1 literature baseline). Turn-granular rather than
    raw-character truncation so the serialisation stays valid JSON and turn IDs
    stay unambiguous; the elision marker records what was dropped."""
    serialised = [json.dumps(asdict(turn), indent=2) for turn in conversation]
    if sum(len(chunk) for chunk in serialised) <= budget_chars:
        kept = list(range(len(conversation)))
    else:
        head: list[int] = []
        tail: list[int] = []
        used = 0
        half = budget_chars / 2
        for index in range(len(conversation)):
            if used + len(serialised[index]) > half:
                break
            head.append(index)
            used += len(serialised[index])
        for index in range(len(conversation) - 1, (head[-1] if head else -1), -1):
            if used + len(serialised[index]) > budget_chars:
                break
            tail.insert(0, index)
            used += len(serialised[index])
        kept = head + tail
    elided = len(conversation) - len(kept)
    items: list[Any] = []
    marker_placed = False
    for position, index in enumerate(kept):
        if (
            not marker_placed
            and position > 0
            and index != kept[position - 1] + 1
        ):
            items.append({"elided_turns": elided})
            marker_placed = True
        items.append(asdict(conversation[index]))
    if elided and not marker_placed:
        items.append({"elided_turns": elided})
    allowed = {
        conversation[index].turn_id
        for index in kept
        if conversation[index].role == "patient"
    }
    return json.dumps(items, indent=2), allowed


def _summary_prompt(case: MediQCase, conversation: tuple[ConversationTurn, ...]) -> str:
    transcript = json.dumps([asdict(turn) for turn in conversation], indent=2)
    return f"""You are a clinical summarizer in a medical multiple-choice research benchmark.
Write one concise free-text summary (at most 200 words) of the patient interview transcript below so that a fresh diagnostic model can answer the benchmark question. Plain prose only: no structure, no fact IDs, no turn citations, no diagnosis, and no information that is not in the transcript.

Benchmark question:
{case.question}

Answer options (context for what matters, not for answering):
{_format_options(case.options)}

Patient transcript:
{transcript}

Return only the required JSON object."""


def _diagnosis_prompt(
    case: MediQCase,
    packet: DiagnosticPacket,
    condition: str,
    conversation: tuple[ConversationTurn, ...] = (),
    summary_text: str = "",
) -> tuple[str, set[str]]:
    supported_conditions = set(DIAGNOSTIC_CONDITIONS) | {
        "full-transcript",
        "truncation-headtail",
        "freetext-summary",
    }
    if condition not in supported_conditions:
        raise ValueError(f"Unknown diagnostic condition: {condition}")
    if condition == "full-transcript":
        if not conversation:
            raise ValueError("Full-transcript condition requires the conversation")
        patient_input = json.dumps([asdict(turn) for turn in conversation], indent=2)
        allowed_patient_ids = {
            turn.turn_id for turn in conversation if turn.role == "patient"
        }
        input_description = "Complete interviewer and patient transcript"
    elif condition == "truncation-headtail":
        if not conversation:
            raise ValueError("Truncation condition requires the conversation")
        budget = len(json.dumps(asdict(packet.handoff), indent=2))
        patient_input, allowed_patient_ids = _truncate_headtail(conversation, budget)
        input_description = (
            "Head-and-tail truncated transcript (middle turns elided to match "
            "the structured-handoff size budget)"
        )
    elif condition == "freetext-summary":
        if not summary_text:
            raise ValueError("Freetext-summary condition requires the summary text")
        patient_input = summary_text
        allowed_patient_ids = {"summary"}
        input_description = (
            "Unstructured free-text interview summary (cite the patient ID "
            '"summary")'
        )
    elif condition == "raw-conversation":
        patient_input = json.dumps([asdict(turn) for turn in packet.source_turns], indent=2)
        allowed_patient_ids = {turn.turn_id for turn in packet.source_turns}
        input_description = "Raw patient turns"
    else:
        patient_input = json.dumps(asdict(packet.handoff), indent=2)
        allowed_patient_ids = {fact.fact_id for fact in packet.handoff.facts}
        input_description = "Structured handoff"
        if condition == "handoff-plus-sources":
            patient_input += "\n\nCited raw patient turns:\n" + json.dumps(
                [asdict(turn) for turn in packet.source_turns], indent=2
            )
            input_description = "Structured handoff plus cited raw patient turns"
    prompt = f"""You are a fresh diagnostic model in a controlled medical multiple-choice benchmark. You did not conduct the interview. Choose exactly one supplied option using the patient input and retrieved textbook evidence.

Question:
{case.question}

Options:
{_format_options(case.options)}

Condition: {condition}
Patient input ({input_description}):
{patient_input}

Retrieved textbook evidence:
{_evidence_text(packet.evidence)}

Rules:
- Return one answer choice from {sorted(case.options)}.
- cited_patient_ids must use only these IDs: {sorted(allowed_patient_ids)}.
- cited_evidence_ids must use only IDs shown in retrieved evidence.
- Cite at least one patient ID and one evidence ID.
- Keep the explanation, differential, and recommendations concise.
- This is benchmark output, not patient-facing medical advice.
Return only the required JSON object."""
    return prompt, allowed_patient_ids


def parse_diagnosis(
    raw: str,
    *,
    options: Mapping[str, str],
    allowed_patient_ids: set[str],
    allowed_evidence_ids: set[str],
) -> dict[str, Any]:
    payload = _json_object(raw)
    choice = str(payload["answer_choice"]).upper()
    if choice not in options:
        raise ValueError(f"Invalid answer choice: {choice}")
    patient_ids = tuple(dict.fromkeys(str(value) for value in payload["cited_patient_ids"]))
    evidence_ids = tuple(dict.fromkeys(str(value) for value in payload["cited_evidence_ids"]))
    if not patient_ids or not evidence_ids:
        raise ValueError("Diagnosis must cite patient information and textbook evidence")
    if not set(patient_ids) <= allowed_patient_ids:
        raise ValueError("Diagnosis cited patient IDs unavailable in this condition")
    if not set(evidence_ids) <= allowed_evidence_ids:
        raise ValueError("Diagnosis cited evidence IDs that were not retrieved")
    confidence = float(payload["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("Diagnosis confidence must be between zero and one")
    return {
        "answer_choice": choice,
        "answer": str(payload["answer"]).strip(),
        "differential": [str(value) for value in payload["differential"]],
        "recommendations": [str(value) for value in payload["recommendations"]],
        "cited_patient_ids": list(patient_ids),
        "cited_evidence_ids": list(evidence_ids),
        "confidence": confidence,
    }


def _invalid_diagnosis_record(
    raw: str, case: MediQCase, response: ProviderResponse, error: Exception
) -> dict[str, Any]:
    prediction = parse_answer_choice(raw, set(case.options))
    return {
        "answer_choice": prediction,
        "answer": "",
        "differential": [],
        "recommendations": [],
        "cited_patient_ids": [],
        "cited_evidence_ids": [],
        "confidence": 0.0,
        "correct": prediction == case.answer_choice,
        "status": "invalid_output",
        "error": f"{type(error).__name__}: {error}",
        "latency_s": response.latency_s,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "checkpoint_reused": response.reused,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _fact_retention(
    handoff: ClinicalHandoff, patient: DeterministicFactPatient
) -> dict[str, Any]:
    revealed = list(patient.revealed.values())
    statements = [fact.statement for fact in handoff.facts]
    recalled = sum(
        any(token_f1(source, statement) >= 0.45 for statement in statements)
        for source in revealed
    )
    supported = sum(
        any(token_f1(statement, source) >= 0.45 for source in revealed)
        for statement in statements
    )
    return {
        "revealed_source_facts": len(revealed),
        "handoff_facts": len(statements),
        "fact_recall": recalled / len(revealed) if revealed else 1.0,
        "fact_precision": supported / len(statements) if statements else 0.0,
    }


def dry_run_plan(
    cases: list[MediQCase],
    spec: Mapping[str, Any],
    model: str,
    diagnostic_conditions: tuple[str, ...] = DIAGNOSTIC_CONDITIONS,
) -> dict[str, Any]:
    max_interviewer_calls = len(cases) * (int(spec["max_questions"]) + 1)
    diagnosis_calls = len(cases) * len(diagnostic_conditions)
    max_calls = max_interviewer_calls + diagnosis_calls
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if not pricing:
        raise ValueError(f"No configured pricing for {model}")
    theoretical_bound = (
        max_calls
        * (
            MAX_PROMPT_CHARS * float(pricing["input"])
            + MAX_OUTPUT_TOKENS * float(pricing["output_including_thinking"])
        )
        / 1_000_000
    )
    return {
        "dataset_fingerprint": dataset_fingerprint(cases, spec),
        "cases": len(cases),
        "max_questions_per_case": spec["max_questions"],
        "max_interviewer_calls": max_interviewer_calls,
        "diagnosis_calls": diagnosis_calls,
        "max_provider_calls": max_calls,
        "patient_model_calls": 0,
        "judge_model_calls": 0,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "theoretical_cost_bound_usd": theoretical_bound,
        "configured_cost_guard_usd": DEFAULT_COST_GUARD_USD,
    }


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _case_run(
    case: MediQCase,
    *,
    provider: Provider,
    retriever: TextbooksBM25Retriever,
    max_questions: int,
    top_k: int,
    diagnostic_conditions: tuple[str, ...] = DIAGNOSTIC_CONDITIONS,
    patient_factory: Callable[[MediQCase], DeterministicFactPatient] = (
        DeterministicFactPatient
    ),
    question_refiner: Any | None = None,
) -> dict[str, Any]:
    if "handoff-plus-sources" not in diagnostic_conditions:
        raise ValueError("Diagnostic conditions require handoff-plus-sources")
    patient = patient_factory(case)
    retrieval_trace: dict[str, Any] = {}
    diagnosis_outputs: dict[str, dict[str, Any]] = {}
    interview_contract_corrections: list[str] = []
    model_trace: list[dict[str, Any]] = []

    def record_response(stage: str, response: ProviderResponse) -> None:
        model_trace.append(
            {
                "stage": stage,
                "latency_s": response.latency_s,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "checkpoint_reused": response.reused,
            }
        )

    def interview(
        conversation: tuple[ConversationTurn, ...], previous: ClinicalHandoff | None
    ) -> InterviewUpdate:
        turn = sum(item.role == "agent" for item in conversation)
        prompt = _interview_prompt(case, conversation, previous, max_questions)
        call_id = f"{case.case_id}:interview:{turn}"
        response = provider(call_id, "interview", prompt, INTERVIEW_SCHEMA)
        record_response("interview", response)
        raw_payload: dict = {}
        last_error: Exception | None = None
        update: InterviewUpdate | None = None
        # Up to 2 retries: first catches json errors (stage name kept for
        # checkpoint compatibility); second catches contract violations such as
        # empty source_turn_ids or unknown patient-turn citations.
        _retry_specs = [
            ("retry-json-v1", "interview-retry-json"),
            ("retry-contract-v1", "interview-retry-contract"),
        ]
        for attempt in range(3):
            if attempt > 0:
                rid, rlabel = _retry_specs[attempt - 1]
                response = provider(
                    f"{call_id}:{rid}",
                    rlabel,
                    prompt,
                    INTERVIEW_SCHEMA,
                )
                record_response(rlabel, response)
            try:
                raw_payload = _json_object(response.raw_output)
                update = parse_interview_update(response.raw_output, conversation)
                last_error = None
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        assert update is not None
        if raw_payload.get("ready_for_diagnosis") and (
            raw_payload.get("missing_information")
            or raw_payload.get("contradictions")
        ):
            interview_contract_corrections.append(
                f"interview:{turn}:ready_normalized_false"
            )
        return update

    def retrieve(handoff: ClinicalHandoff) -> tuple[EvidenceItem, ...]:
        query = " ".join([case.question, *(fact.statement for fact in handoff.facts)])
        started = time.perf_counter()
        snippets = retriever.retrieve(query, k=top_k)
        retrieval_trace.update(
            {
                "query": query,
                "options_excluded": True,
                "latency_s": time.perf_counter() - started,
                "retrieved_ids": [snippet.snippet_id for snippet in snippets],
                "context_count": len(snippets),
            }
        )
        return tuple(
            EvidenceItem(
                evidence_id=snippet.snippet_id,
                text=snippet.content,
                source=snippet.title,
            )
            for snippet in snippets
        )

    def diagnose_condition(
        packet: DiagnosticPacket,
        condition: str,
        conversation: tuple[ConversationTurn, ...] = (),
        summary_text: str = "",
    ) -> tuple[dict[str, Any] | None, ProviderResponse, Exception | None]:
        prompt, allowed_patient_ids = _diagnosis_prompt(
            case, packet, condition, conversation=conversation, summary_text=summary_text
        )
        call_id = f"{case.case_id}:diagnose:{condition}"
        stage = f"diagnose:{condition}"
        attempts = [provider(call_id, stage, prompt, diagnosis_schema(case.options))]
        record_response(stage, attempts[-1])

        def parse(response: ProviderResponse) -> dict[str, Any]:
            return parse_diagnosis(
                response.raw_output,
                options=case.options,
                allowed_patient_ids=allowed_patient_ids,
                allowed_evidence_ids={item.evidence_id for item in packet.evidence},
            )

        parsed: dict[str, Any] | None = None
        final_error: Exception | None = None
        try:
            parsed = parse(attempts[-1])
        except Exception as error:
            retryable = isinstance(error, json.JSONDecodeError) or str(error).startswith(
                "Invalid answer choice:"
            )
            if retryable:
                # Up to 2 retries for transient truncation or contract errors.
                for retry_n in range(1, 3):
                    retry_stage = f"{stage}:retry-contract-v{retry_n}"
                    attempts.append(
                        provider(
                            f"{call_id}:retry-contract-v{retry_n}",
                            retry_stage,
                            prompt,
                            diagnosis_schema(case.options),
                        )
                    )
                    record_response(retry_stage, attempts[-1])
                    try:
                        parsed = parse(attempts[-1])
                        final_error = None
                        break
                    except Exception as retry_error:
                        final_error = retry_error
            else:
                final_error = error

        combined = ProviderResponse(
            raw_output=attempts[-1].raw_output,
            input_tokens=sum(item.input_tokens for item in attempts),
            output_tokens=sum(item.output_tokens for item in attempts),
            total_tokens=sum(item.total_tokens for item in attempts),
            latency_s=sum(item.latency_s for item in attempts),
            reused=all(item.reused for item in attempts),
        )
        return parsed, combined, final_error

    def diagnose_primary(packet: DiagnosticPacket) -> DiagnosticDraft:
        condition = "handoff-plus-sources"
        parsed, response, error = diagnose_condition(packet, condition)
        if error is not None:
            diagnosis_outputs[condition] = _invalid_diagnosis_record(
                response.raw_output, case, response, error
            )
            raise error
        assert parsed is not None
        diagnosis_outputs[condition] = {
            **parsed,
            "correct": parsed["answer_choice"] == case.answer_choice,
            "status": "completed",
            "error": "",
            "latency_s": response.latency_s,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "checkpoint_reused": response.reused,
        }
        return DiagnosticDraft(
            answer=f"Final answer: {parsed['answer_choice']}. {parsed['answer']}",
            differential=tuple(parsed["differential"]),
            recommendations=tuple(parsed["recommendations"]),
            cited_fact_ids=tuple(parsed["cited_patient_ids"]),
            cited_evidence_ids=tuple(parsed["cited_evidence_ids"]),
            confidence=float(parsed["confidence"]),
        )

    def validate(_packet: DiagnosticPacket, draft: DiagnosticDraft) -> SafetyDecision:
        choice = parse_answer_choice(draft.answer, set(case.options))
        if choice in case.options:
            return SafetyDecision("approve", "Valid benchmark choice with checked citations.")
        return SafetyDecision("abstain", "The diagnostic output has no valid benchmark choice.")

    graph = build_clinical_handoff_agent(
        interview=interview,
        tools=ClinicalTools(
            ask_patient=patient,
            retrieve_evidence=retrieve,
            refine_question=question_refiner,
        ),
        diagnose=diagnose_primary,
        validate_safety=validate,
    )
    state = graph.invoke(
        initial_clinical_state(
            case.initial_info,
            max_questions=max_questions,
            include_source_turns=True,
            force_diagnosis_on_budget=True,
            allow_incomplete_finalize=True,
        ),
        config={"recursion_limit": 30},
    )
    freetext_summary = ""
    if state.get("diagnostic_packet"):
        packet = state["diagnostic_packet"]
        conversation = tuple(state.get("conversation", []))
        for condition in diagnostic_conditions:
            if condition == "handoff-plus-sources":
                continue
            condition_packet = packet
            if condition == "raw-conversation":
                condition_packet = DiagnosticPacket(
                    handoff=packet.handoff,
                    evidence=packet.evidence,
                    source_turns=tuple(
                        turn
                        for turn in state.get("conversation", [])
                        if turn.role == "patient"
                    ),
                )
            summary_text = ""
            if condition == "freetext-summary":
                summary_response = provider(
                    f"{case.case_id}:summarize",
                    "summarize",
                    _summary_prompt(case, conversation),
                    SUMMARY_SCHEMA,
                )
                record_response("summarize", summary_response)
                summary_text = str(
                    _json_object(summary_response.raw_output).get("summary", "")
                ).strip()
                freetext_summary = summary_text
            response: ProviderResponse | None = None
            try:
                parsed, response, error = diagnose_condition(
                    condition_packet,
                    condition,
                    conversation=conversation,
                    summary_text=summary_text,
                )
                if error is not None:
                    raise error
                assert parsed is not None
                diagnosis_outputs[condition] = {
                    **parsed,
                    "correct": parsed["answer_choice"] == case.answer_choice,
                    "status": "completed",
                    "error": "",
                    "latency_s": response.latency_s,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "checkpoint_reused": response.reused,
                }
            except Exception as error:
                if response is None:
                    raise
                diagnosis_outputs[condition] = _invalid_diagnosis_record(
                    response.raw_output, case, response, error
                )

    handoff = state.get("handoff")
    interview_latency = sum(
        item["latency_s"]
        for item in model_trace
        if item["stage"].startswith("interview")
    )
    pipeline_latency_by_condition = {
        condition: (
            interview_latency
            + float(retrieval_trace.get("latency_s", 0.0))
            + float(diagnosis_outputs[condition]["latency_s"])
        )
        for condition in diagnosis_outputs
    }
    return {
        "case_id": case.case_id,
        "source_id": case.source_id,
        "specialty": case.specialty,
        "gold_choice": case.answer_choice,
        "status": state.get("status"),
        "action": state.get("action"),
        "questions_asked": int(state.get("questions_asked", 0)),
        "interview_calls": int(state.get("interview_calls", 0)),
        "patient_tool_calls": int(state.get("patient_tool_calls", 0)),
        "patient_matcher_version": patient.matcher_version,
        "question_refiner_version": getattr(question_refiner, "version", "none"),
        "retrieval": retrieval_trace,
        "handoff": asdict(handoff) if handoff else None,
        "freetext_summary": freetext_summary,
        "handoff_metrics": _fact_retention(handoff, patient) if handoff else None,
        "interview_contract_corrections": interview_contract_corrections,
        "revealed_patient_facts": patient.revealed,
        "conversation": [asdict(turn) for turn in state.get("conversation", [])],
        "diagnoses": diagnosis_outputs,
        "model_trace": model_trace,
        "pipeline_latency_by_condition": pipeline_latency_by_condition,
        "errors": list(state.get("errors", [])),
        "trace": list(state.get("trace", [])),
    }


def summarize(
    results: list[Mapping[str, Any]],
    provider: CheckpointedGeminiProvider,
    diagnostic_conditions: tuple[str, ...] = DIAGNOSTIC_CONDITIONS,
    primary_condition: str = "handoff-plus-sources",
) -> dict[str, Any]:
    calls = list(provider.payload["calls"].values())
    interview_calls = [
        item for item in calls if item["stage"].startswith("interview")
    ]
    shared_interview_input = sum(int(item["input_tokens"]) for item in interview_calls)
    shared_interview_output = sum(int(item["output_tokens"]) for item in interview_calls)
    shared_interview_cost = provider.estimate_cost(
        shared_interview_input, shared_interview_output
    )
    by_condition: dict[str, Any] = {}
    for condition in diagnostic_conditions:
        outputs = [
            result["diagnoses"][condition]
            for result in results
            if condition in result.get("diagnoses", {})
        ]
        diagnosis_input = sum(int(item["input_tokens"]) for item in outputs)
        diagnosis_output = sum(int(item["output_tokens"]) for item in outputs)
        valid_outputs = [item for item in outputs if item.get("status") == "completed"]
        valid_correct = sum(bool(item["correct"]) for item in valid_outputs)
        pipeline_latencies = [
            float(result["pipeline_latency_by_condition"][condition])
            for result in results
            if condition in result.get("pipeline_latency_by_condition", {})
        ]
        by_condition[condition] = {
            "completed": len(outputs),
            "correct": sum(bool(item["correct"]) for item in outputs),
            "accuracy": (
                sum(bool(item["correct"]) for item in outputs) / len(outputs)
                if outputs
                else 0.0
            ),
            "accuracy_95ci": _wilson_interval(
                sum(bool(item["correct"]) for item in outputs), len(outputs)
            ),
            "valid_outputs": len(valid_outputs),
            "valid_output_rate": (
                len(valid_outputs) / len(outputs)
                if outputs
                else 0.0
            ),
            "selective_accuracy": (
                valid_correct / len(valid_outputs) if valid_outputs else 0.0
            ),
            "invalid_but_exact_choice_correct": sum(
                item.get("status") != "completed" and bool(item["correct"])
                for item in outputs
            ),
            "invalid_reason_counts": dict(
                Counter(
                    item["error"].split(":", 1)[-1].strip()
                    for item in outputs
                    if item.get("status") != "completed"
                )
            ),
            "accuracy_by_specialty": {
                specialty: {
                    "correct": sum(
                        bool(result["diagnoses"][condition]["correct"])
                        for result in results
                        if result.get("specialty") == specialty
                        and condition in result.get("diagnoses", {})
                    ),
                    "total": sum(
                        1
                        for result in results
                        if result.get("specialty") == specialty
                        and condition in result.get("diagnoses", {})
                    ),
                }
                for specialty in sorted(
                    {str(result.get("specialty")) for result in results}
                )
            },
            "p50_latency_s": statistics.median([item["latency_s"] for item in outputs]) if outputs else 0.0,
            "p95_latency_s": _percentile([item["latency_s"] for item in outputs], 0.95),
            "diagnosis_input_tokens": diagnosis_input,
            "diagnosis_output_tokens": diagnosis_output,
            "diagnosis_cost_usd": provider.estimate_cost(
                diagnosis_input, diagnosis_output
            ),
            "conceptual_pipeline_input_tokens": (
                shared_interview_input + diagnosis_input
            ),
            "conceptual_pipeline_output_tokens": (
                shared_interview_output + diagnosis_output
            ),
            "conceptual_pipeline_cost_usd": (
                shared_interview_cost
                + provider.estimate_cost(diagnosis_input, diagnosis_output)
            ),
            "pipeline_p50_latency_s": (
                statistics.median(pipeline_latencies) if pipeline_latencies else 0.0
            ),
            "pipeline_p95_latency_s": _percentile(pipeline_latencies, 0.95),
        }
    handoff_metrics = [
        result["handoff_metrics"] for result in results if result.get("handoff_metrics")
    ]
    retrievals = [result["retrieval"] for result in results if result.get("retrieval")]
    stages: dict[str, Any] = {}
    for stage in sorted({item["stage"] for item in calls}):
        stage_calls = [item for item in calls if item["stage"] == stage]
        latencies = [float(item["latency_s"]) for item in stage_calls]
        input_tokens = sum(int(item["input_tokens"]) for item in stage_calls)
        output_tokens = sum(int(item["output_tokens"]) for item in stage_calls)
        stages[stage] = {
            "calls": len(stage_calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "p50_latency_s": statistics.median(latencies),
            "p95_latency_s": _percentile(latencies, 0.95),
            "estimated_cost_usd": provider.estimate_cost(input_tokens, output_tokens),
        }

    paired: dict[str, Any] = {}
    for baseline in (
        condition
        for condition in diagnostic_conditions
        if condition != primary_condition
    ):
        left_wins = right_wins = ties = 0
        for result in results:
            diagnoses = result.get("diagnoses", {})
            if primary_condition not in diagnoses or baseline not in diagnoses:
                continue
            left = bool(diagnoses[primary_condition]["correct"])
            right = bool(diagnoses[baseline]["correct"])
            if left and not right:
                left_wins += 1
            elif right and not left:
                right_wins += 1
            else:
                ties += 1
        paired[f"{primary_condition}_vs_{baseline}"] = {
            "primary_wins": left_wins,
            "baseline_wins": right_wins,
            "ties": ties,
            "mcnemar_exact_p": _mcnemar_exact_p(left_wins, right_wins),
        }
    return {
        "cases_completed": len(results),
        "mean_questions": statistics.mean([result["questions_asked"] for result in results]) if results else 0.0,
        "question_budget_rate": (
            sum(result["questions_asked"] == 3 for result in results) / len(results)
            if results
            else 0.0
        ),
        "mean_lexical_handoff_fact_recall": statistics.mean([item["fact_recall"] for item in handoff_metrics]) if handoff_metrics else 0.0,
        "mean_lexical_handoff_fact_precision": statistics.mean([item["fact_precision"] for item in handoff_metrics]) if handoff_metrics else 0.0,
        "interview_contract_corrections": sum(
            len(result.get("interview_contract_corrections", []))
            for result in results
        ),
        "question_refinement_cases": sum(
            "question_refined" in result.get("trace", []) for result in results
        ),
        "question_refinement_case_rate": (
            sum("question_refined" in result.get("trace", []) for result in results)
            / len(results)
            if results
            else 0.0
        ),
        "question_refinements": sum(
            result.get("trace", []).count("question_refined") for result in results
        ),
        "patient_tool_success_rate": (
            sum(result.get("trace", []).count("patient_tool_success") for result in results)
            / sum(int(result.get("patient_tool_calls", 0)) for result in results)
            if sum(int(result.get("patient_tool_calls", 0)) for result in results)
            else 0.0
        ),
        "retrieval_empty_rate": (
            sum(item.get("context_count", 0) == 0 for item in retrievals) / len(retrievals)
            if retrievals
            else 0.0
        ),
        "mean_retrieval_latency_s": statistics.mean([item["latency_s"] for item in retrievals]) if retrievals else 0.0,
        "diagnostic_conditions": by_condition,
        "paired": paired,
        "provider": {
            "logical_calls": len(calls),
            "input_tokens": sum(int(item["input_tokens"]) for item in calls),
            "output_tokens": sum(int(item["output_tokens"]) for item in calls),
            "total_tokens": sum(int(item["total_tokens"]) for item in calls),
            "p50_latency_s": statistics.median([float(item["latency_s"]) for item in calls]) if calls else 0.0,
            "p95_latency_s": _percentile([float(item["latency_s"]) for item in calls], 0.95),
            "estimated_cost_usd": provider.spent,
            "retry_calls": sum("retry" in item["stage"] for item in calls),
            "provider_errors": 0,
            "provider_error_rate": 0.0,
            "stages": stages,
        },
    }


def run_pilot(
    *,
    cases: list[MediQCase],
    spec: Mapping[str, Any],
    model: str,
    index_path: Path,
    provider_checkpoint: Path,
    output_path: Path,
    cost_guard_usd: float = DEFAULT_COST_GUARD_USD,
    reuse_only: bool = False,
) -> dict[str, Any]:
    fingerprint = dataset_fingerprint(cases, spec)
    provider = CheckpointedGeminiProvider(
        model=model,
        checkpoint_path=provider_checkpoint,
        fingerprint=fingerprint,
        max_calls=MAX_PROVIDER_CALLS,
        max_cost_usd=cost_guard_usd,
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_new_calls=not reuse_only,
    )
    retriever = TextbooksBM25Retriever(index_path)
    results: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        result = _case_run(
            case,
            provider=provider,
            retriever=retriever,
            max_questions=int(spec["max_questions"]),
            top_k=int(spec["retrieval"]["top_k"]),
        )
        results.append(result)
        report = {
            "format_version": 1,
            "stage": "5N",
            "benchmark": "MediQ handoff pilot",
            "dataset_fingerprint": fingerprint,
            "source_revision": spec["revision"],
            "source_sha256": spec["source_sha256"],
            "model": model,
            "cost_guard_usd": cost_guard_usd,
            "results": results,
            "metrics": summarize(results, provider),
        }
        _save_json(output_path, report)
        print(
            f"[{position}/{len(cases)}] {case.case_id}: {result['status']} "
            f"questions={result['questions_asked']} cost=${provider.spent:.4f}"
        )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--provider-checkpoint", type=Path, default=DEFAULT_PROVIDER_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--cost-guard", type=float, default=DEFAULT_COST_GUARD_USD)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Fail instead of calling Gemini when any checkpoint entry is missing",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    if args.download:
        download_source(args.source, spec)
    if not args.source.exists():
        raise FileNotFoundError("Pinned MediQ source is missing; rerun with --download")
    cases = load_cases(args.source, spec)
    plan = dry_run_plan(cases, spec, args.model)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if plan["max_provider_calls"] > MAX_PROVIDER_CALLS:
        raise RuntimeError("Dry-run provider call count exceeds the hard guard")
    if plan["theoretical_cost_bound_usd"] > args.cost_guard:
        raise RuntimeError("Dry-run theoretical cost exceeds the configured guard")
    report = run_pilot(
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
