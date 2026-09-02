"""
Phase IG — Information-Gain Guided Interviewing.

Three arms share the same diagnostic representation (structured-handoff):
  llm-freeform   existing interview prompt, single question per turn
  oracle-ig      M candidates → query real patient tool → pick max ΔH
  simulated-ig   M candidates → imagined answers in same call → pick max ΔH

All provider calls are checkpointed. The runner is invoked via:
    python -m graphrag.eval.mediq_ig --set dev|holdout [--limit N] [--dry-run]

Spec: graphrag/eval/specs/mediq_ig_selection.json
Design: docs/specs/information_gain_interviewing.md
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import statistics
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from graphrag.agent.clinical_handoff import (
    ClinicalHandoff,
    ConversationTurn,
    DiagnosticPacket,
    EvidenceItem,
)
from graphrag.eval.checkpointed_gemini import CheckpointedGeminiProvider, ProviderResponse
from graphrag.eval.mediq_handoff_benchmark import (
    _diagnosis_prompt,
    _json_object,
    diagnosis_schema,
    parse_answer_choice,
    parse_diagnosis,
    parse_interview_update,
    _interview_prompt,
    INTERVIEW_SCHEMA,
    token_f1,
)
from graphrag.eval.mediq_handoff_data import (
    ConceptAwareFactPatient,
    DEFAULT_ROOT,
    DeterministicFactPatient,
    MediQCase,
    dataset_fingerprint,
    load_cases,
)
from graphrag.eval.mirage_benchmark import (
    MODEL_PRICING_USD_PER_MILLION,
    _mcnemar_exact_p,
    _wilson_interval,
)
from graphrag.eval.mirage_corpus import DEFAULT_INDEX, TextbooksBM25Retriever

IG_SPEC_PATH = pathlib.Path("graphrag/eval/specs/mediq_ig_selection.json")
IG_CHECKPOINT = DEFAULT_ROOT / "ig_provider_checkpoint.json"
IG_RESULTS = DEFAULT_ROOT / "ig_results.json"
IG_LOG = DEFAULT_ROOT / "ig_run_log.jsonl"

N_CANDIDATES = 3
MAX_QUESTIONS = 3
TOP_K = 8
MAX_OUTPUT_TOKENS = 2_048
MAX_PROVIDER_CALLS = 5_000
MAX_PROMPT_CHARS = 16_000

ARMS = ("llm-freeform", "oracle-ig", "simulated-ig")

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

IG_CANDIDATE_SCHEMA_ORACLE = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "target_information": {"type": "string"},
                },
                "required": ["question", "target_information"],
            },
        }
    },
    "required": ["candidates"],
}

IG_CANDIDATE_SCHEMA_SIMULATED = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "target_information": {"type": "string"},
                    "simulated_answer": {"type": "string"},
                },
                "required": ["question", "target_information", "simulated_answer"],
            },
        }
    },
    "required": ["candidates"],
}

IG_SCORING_SCHEMA_TMPL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "option_probabilities": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "most_likely_choice": {"type": "string"},
    },
    "required": ["option_probabilities", "most_likely_choice"],
}


def ig_scoring_schema(options: Mapping[str, str]) -> dict[str, Any]:
    schema = copy.deepcopy(IG_SCORING_SCHEMA_TMPL)
    schema["properties"]["most_likely_choice"]["enum"] = sorted(options)
    return schema


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def _format_options(options: Mapping[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def _candidate_prompt_oracle(
    case: MediQCase,
    conversation: tuple[ConversationTurn, ...],
    handoff: ClinicalHandoff,
    n_candidates: int,
) -> str:
    questions_used = sum(t.role == "agent" for t in conversation)
    return f"""You are the interviewing component of a medical benchmark.
Generate {n_candidates} distinct candidate questions to ask the patient next.
Each should target different potentially discriminative information for choosing among the answer options.

Question:
{case.question}

Options:
{_format_options(case.options)}

Questions used: {questions_used}/{MAX_QUESTIONS}

Current handoff:
{json.dumps(asdict(handoff), indent=2)}

Return {n_candidates} distinct questions. Do not overlap with information already in the handoff.
Return only the required JSON object."""


def _candidate_prompt_simulated(
    case: MediQCase,
    conversation: tuple[ConversationTurn, ...],
    handoff: ClinicalHandoff,
    n_candidates: int,
) -> str:
    questions_used = sum(t.role == "agent" for t in conversation)
    return f"""You are the interviewing component of a medical benchmark.
Generate {n_candidates} distinct candidate questions and a plausible patient answer for each.

Question:
{case.question}

Options:
{_format_options(case.options)}

Questions used: {questions_used}/{MAX_QUESTIONS}

Current handoff:
{json.dumps(asdict(handoff), indent=2)}

For each candidate: write a concise, realistic patient answer (1-2 sentences) as if
the patient answered that question. Base the simulated answer on what a typical patient
with this presentation might say; do not access hidden facts.
Return only the required JSON object."""


def _scoring_prompt(
    case: MediQCase,
    handoff: ClinicalHandoff,
    evidence: tuple[EvidenceItem, ...],
    question: str,
    answer: str,
) -> str:
    evidence_text = "\n".join(
        f"[{item.evidence_id}] {item.source}: {item.text}" for item in evidence
    )
    return f"""You are a diagnostic scoring function for a medical benchmark.
Given the current clinical handoff PLUS a hypothetical question-answer pair,
return a probability distribution over the answer choices.
This is HYPOTHETICAL — you are estimating what you would believe after seeing this answer.

Question:
{case.question}

Options:
{_format_options(case.options)}

Current handoff (before the new Q&A):
{json.dumps(asdict(handoff), indent=2)}

Hypothetical Q&A:
  Q: {question}
  A: {answer}

Retrieved textbook evidence:
{evidence_text}

Return a probability distribution that sums to 1.0 over the listed options.
Be calibrated: if you are uncertain, spread probability across options.
Return only the required JSON object."""


def _baseline_scoring_prompt(
    case: MediQCase,
    handoff: ClinicalHandoff,
    evidence: tuple[EvidenceItem, ...],
) -> str:
    evidence_text = "\n".join(
        f"[{item.evidence_id}] {item.source}: {item.text}" for item in evidence
    )
    return f"""You are a diagnostic scoring function for a medical benchmark.
Return a probability distribution over the answer choices given the current handoff.

Question:
{case.question}

Options:
{_format_options(case.options)}

Current handoff:
{json.dumps(asdict(handoff), indent=2)}

Retrieved textbook evidence:
{evidence_text}

Return only the required JSON object."""


# --------------------------------------------------------------------------
# Entropy helpers
# --------------------------------------------------------------------------

def _entropy(probs: dict[str, float]) -> float:
    total = sum(probs.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs.values():
        q = p / total
        if q > 0:
            h -= q * math.log(q)
    return h


def _parse_probs(raw: str, options: Mapping[str, str]) -> dict[str, float]:
    payload = _json_object(raw)
    raw_probs = payload.get("option_probabilities", {})
    keys = sorted(options)
    probs = {k: max(0.0, float(raw_probs.get(k, 0.0))) for k in keys}
    total = sum(probs.values())
    if total <= 0:
        # Uniform fallback
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in probs.items()}


def _delta_h(h_before: float, h_after: float) -> float:
    return h_before - h_after  # positive = information gained


# --------------------------------------------------------------------------
# Per-arm interview loops
# --------------------------------------------------------------------------

def _run_llm_freeform_arm(
    case: MediQCase,
    provider: Callable,
    patient: Callable,
    retriever: TextbooksBM25Retriever,
    record: Callable,
) -> dict[str, Any]:
    """Runs the existing freeform interview logic, returns interview state."""
    conversation: tuple[ConversationTurn, ...] = (
        ConversationTurn("patient-0", "patient", case.initial_info),
    )
    handoff: ClinicalHandoff | None = None
    errors: list[str] = []
    arm_calls: list[dict] = []

    def prov(call_id: str, stage: str, prompt: str, schema: dict) -> ProviderResponse:
        r = provider(call_id, stage, prompt, schema)
        record(stage, r)
        arm_calls.append({"call_id": call_id, "stage": stage,
                          "input_tokens": r.input_tokens, "output_tokens": r.output_tokens})
        return r

    for turn_n in range(MAX_QUESTIONS):
        call_id = f"{case.case_id}:ff:interview:{turn_n}"
        prompt = _interview_prompt(case, conversation, handoff, MAX_QUESTIONS)
        last_error = None
        update = None
        for attempt, (rid, rlabel) in enumerate([
            (None, "ff-interview"),
            ("retry-json-v1", "ff-interview-retry-json"),
            ("retry-contract-v1", "ff-interview-retry-contract"),
        ]):
            if attempt == 0:
                response = prov(call_id, rlabel, prompt, INTERVIEW_SCHEMA)
            else:
                response = prov(f"{call_id}:{rid}", rlabel, prompt, INTERVIEW_SCHEMA)
            try:
                _json_object(response.raw_output)
                update = parse_interview_update(response.raw_output, conversation)
                last_error = None
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            errors.append(f"interview_error:{last_error}")
            break

        handoff = update.handoff
        if update.decision.action == "finalize":
            break
        question = update.decision.question
        if not question:
            break
        patient_answer = patient(question, conversation)
        agent_turn_id = f"agent-{turn_n + 1}"
        patient_turn_id = f"patient-{turn_n + 1}"
        conversation = conversation + (
            ConversationTurn(agent_turn_id, "agent", question),
            ConversationTurn(patient_turn_id, "patient", patient_answer),
        )

    questions_asked = sum(t.role == "agent" for t in conversation)
    return {
        "conversation": conversation,
        "handoff": handoff,
        "questions_asked": questions_asked,
        "arm_calls": arm_calls,
        "errors": errors,
    }


def _run_ig_arm(
    case: MediQCase,
    provider: Callable,
    patient: Callable,
    retriever: TextbooksBM25Retriever,
    record: Callable,
    mode: str,  # "oracle" or "simulated"
) -> dict[str, Any]:
    """Runs oracle-ig or simulated-ig arm."""
    conversation: tuple[ConversationTurn, ...] = (
        ConversationTurn("patient-0", "patient", case.initial_info),
    )
    handoff: ClinicalHandoff | None = None
    errors: list[str] = []
    arm_calls: list[dict] = []
    ig_trace: list[dict] = []

    prefix = "or" if mode == "oracle" else "si"

    def prov(call_id: str, stage: str, prompt: str, schema: dict) -> ProviderResponse:
        r = provider(call_id, stage, prompt, schema)
        record(stage, r)
        arm_calls.append({"call_id": call_id, "stage": stage,
                          "input_tokens": r.input_tokens, "output_tokens": r.output_tokens})
        return r

    # We need evidence for scoring — retrieve once per case (same as P1 path).
    # Evidence is retrieved based on the current question + handoff facts.
    def retrieve(h: ClinicalHandoff) -> tuple[EvidenceItem, ...]:
        query = " ".join([case.question] + (
            [fact.statement for fact in h.facts] if h else []
        ))
        snippets = retriever.retrieve(query, k=TOP_K)
        return tuple(
            EvidenceItem(s.snippet_id, s.content, s.title) for s in snippets
        )

    for turn_n in range(MAX_QUESTIONS):
        cand_schema = (
            IG_CANDIDATE_SCHEMA_SIMULATED if mode == "simulated"
            else IG_CANDIDATE_SCHEMA_ORACLE
        )
        cand_prompt = (
            _candidate_prompt_simulated(case, conversation, handoff or _empty_handoff(case), N_CANDIDATES)
            if mode == "simulated"
            else _candidate_prompt_oracle(case, conversation, handoff or _empty_handoff(case), N_CANDIDATES)
        )
        cand_call_id = f"{case.case_id}:{prefix}:cand:{turn_n}"
        cand_resp = prov(cand_call_id, f"{prefix}-candidates", cand_prompt, cand_schema)

        try:
            cand_payload = _json_object(cand_resp.raw_output)
            candidates = cand_payload.get("candidates", [])
        except (json.JSONDecodeError, ValueError):
            errors.append(f"candidate_gen_error:turn_{turn_n}")
            break
        if not candidates:
            errors.append(f"no_candidates:turn_{turn_n}")
            break

        evidence = retrieve(handoff or _empty_handoff(case))

        # Baseline entropy H(P(answer | H_t))
        baseline_prompt = _baseline_scoring_prompt(case, handoff or _empty_handoff(case), evidence)
        baseline_resp = prov(
            f"{case.case_id}:{prefix}:baseline:{turn_n}",
            f"{prefix}-baseline",
            baseline_prompt,
            ig_scoring_schema(case.options),
        )
        try:
            h_before = _entropy(_parse_probs(baseline_resp.raw_output, case.options))
        except Exception:
            h_before = math.log(len(case.options))  # max entropy fallback

        # Score each candidate
        best_idx = 0
        best_delta = -1.0
        scored_candidates = []
        for m, cand in enumerate(candidates[:N_CANDIDATES]):
            question = str(cand.get("question", "")).strip()
            if not question:
                continue

            # Get patient answer
            if mode == "oracle":
                patient_answer = patient(question, conversation)
            else:
                patient_answer = str(cand.get("simulated_answer", "")).strip() or "I don't know."

            # Score: P(answer | H + Q&A)
            score_prompt = _scoring_prompt(
                case, handoff or _empty_handoff(case), evidence, question, patient_answer
            )
            score_resp = prov(
                f"{case.case_id}:{prefix}:score:{turn_n}:{m}",
                f"{prefix}-score",
                score_prompt,
                ig_scoring_schema(case.options),
            )
            try:
                h_after = _entropy(_parse_probs(score_resp.raw_output, case.options))
            except Exception:
                h_after = h_before

            delta = _delta_h(h_before, h_after)
            scored_candidates.append({
                "question": question,
                "patient_answer": patient_answer,
                "delta_h": delta,
                "h_after": h_after,
            })
            if delta > best_delta:
                best_delta = delta
                best_idx = m

        if not scored_candidates:
            errors.append(f"no_scored_candidates:turn_{turn_n}")
            break

        selected = scored_candidates[best_idx]
        ig_trace.append({
            "turn": turn_n,
            "h_before": h_before,
            "selected_delta_h": selected["delta_h"],
            "candidates": scored_candidates,
        })

        # Now run the actual interview with the selected question (updates handoff)
        # We inject the selected question as a pre-committed choice and let the
        # interviewer update the handoff with the real patient answer.
        best_question = selected["question"]
        real_answer = (
            selected["patient_answer"] if mode == "oracle"
            else patient(best_question, conversation)
        )

        agent_turn_id = f"agent-{turn_n + 1}"
        patient_turn_id = f"patient-{turn_n + 1}"
        conversation = conversation + (
            ConversationTurn(agent_turn_id, "agent", best_question),
            ConversationTurn(patient_turn_id, "patient", real_answer),
        )

        # Update handoff via the standard interviewer (processes the new Q&A into facts)
        update_prompt = _interview_prompt(case, conversation, handoff, MAX_QUESTIONS)
        update_id = f"{case.case_id}:{prefix}:update:{turn_n}"
        last_error = None
        for attempt, (rid, rlabel) in enumerate([
            (None, f"{prefix}-update"),
            ("retry-json-v1", f"{prefix}-update-retry-json"),
            ("retry-contract-v1", f"{prefix}-update-retry-contract"),
        ]):
            if attempt == 0:
                resp = prov(update_id, rlabel, update_prompt, INTERVIEW_SCHEMA)
            else:
                resp = prov(f"{update_id}:{rid}", rlabel, update_prompt, INTERVIEW_SCHEMA)
            try:
                _json_object(resp.raw_output)
                update = parse_interview_update(resp.raw_output, conversation)
                last_error = None
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        if last_error is not None:
            errors.append(f"interview_update_error:turn_{turn_n}:{last_error}")
            break

        handoff = update.handoff
        if update.decision.action == "finalize":
            break

    questions_asked = sum(t.role == "agent" for t in conversation)
    return {
        "conversation": conversation,
        "handoff": handoff,
        "questions_asked": questions_asked,
        "arm_calls": arm_calls,
        "ig_trace": ig_trace,
        "errors": errors,
    }


def _empty_handoff(case: MediQCase) -> ClinicalHandoff:
    return ClinicalHandoff(chief_complaint=case.initial_info, facts=())


# --------------------------------------------------------------------------
# Per-case runner
# --------------------------------------------------------------------------

def _diagnose(
    case: MediQCase,
    handoff: ClinicalHandoff | None,
    conversation: tuple[ConversationTurn, ...],
    evidence: tuple[EvidenceItem, ...],
    provider: Callable,
    record: Callable,
    arm: str,
) -> dict[str, Any]:
    if handoff is None:
        return {
            "answer_choice": None,
            "correct": False,
            "status": "skipped",
            "error": "no_handoff",
            "confidence": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    packet = DiagnosticPacket(
        handoff=handoff,
        evidence=evidence,
        source_turns=tuple(t for t in conversation if t.role == "patient"),
    )
    prompt, allowed_ids = _diagnosis_prompt(case, packet, "structured-handoff")
    call_id = f"{case.case_id}:{arm}:diagnose"
    parsed = None
    final_error = None
    for retry_n in range(3):
        suffix = "" if retry_n == 0 else f":retry-v{retry_n}"
        stage = f"{arm}-diagnose{suffix}"
        resp = provider(call_id + suffix, stage, prompt, diagnosis_schema(case.options))
        record(stage, resp)
        try:
            parsed = parse_diagnosis(
                resp.raw_output,
                options=case.options,
                allowed_patient_ids=allowed_ids,
                allowed_evidence_ids={item.evidence_id for item in evidence},
            )
            final_error = None
            break
        except Exception as exc:
            final_error = exc

    if final_error is not None or parsed is None:
        return {
            "answer_choice": None,
            "correct": False,
            "status": "error",
            "error": str(final_error),
            "confidence": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    return {
        **parsed,
        "correct": parsed["answer_choice"] == case.answer_choice,
        "status": "completed",
        "error": "",
    }


def run_ig_case(
    case: MediQCase,
    provider: Callable,
    retriever: TextbooksBM25Retriever,
) -> dict[str, Any]:
    model_trace: list[dict] = []

    def record(stage: str, r: ProviderResponse) -> None:
        model_trace.append({
            "stage": stage,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_s": r.latency_s,
            "checkpoint_reused": r.reused,
        })

    patient_oracle = ConceptAwareFactPatient(case)
    patient_freeform = DeterministicFactPatient(case)

    results_by_arm: dict[str, dict] = {}

    for arm in ARMS:
        if arm == "llm-freeform":
            state = _run_llm_freeform_arm(case, provider, patient_freeform, retriever, record)
        elif arm == "oracle-ig":
            state = _run_ig_arm(case, provider, patient_oracle, retriever, record, mode="oracle")
        else:
            state = _run_ig_arm(case, provider, patient_oracle, retriever, record, mode="simulated")

        handoff = state["handoff"]
        conversation = state["conversation"]

        # Retrieve evidence (per-arm, since handoff may differ)
        if handoff:
            query = " ".join([case.question] + [f.statement for f in handoff.facts])
        else:
            query = case.question
        snippets = retriever.retrieve(query, k=TOP_K)
        evidence = tuple(
            EvidenceItem(s.snippet_id, s.content, s.title) for s in snippets
        )

        diag = _diagnose(case, handoff, conversation, evidence, provider, record, arm)
        results_by_arm[arm] = {
            "questions_asked": state["questions_asked"],
            "errors": state["errors"],
            "diagnosis": diag,
            "ig_trace": state.get("ig_trace", []),
        }

    overall_status = "answered"
    for arm_data in results_by_arm.values():
        if arm_data["errors"]:
            overall_status = "interview_error"
            break

    return {
        "case_id": case.case_id,
        "source_id": case.source_id,
        "specialty": case.specialty,
        "gold_choice": case.answer_choice,
        "status": overall_status,
        "arms": results_by_arm,
        "model_trace": model_trace,
    }


# --------------------------------------------------------------------------
# Dataset / spec loading
# --------------------------------------------------------------------------

def load_ig_spec() -> dict[str, Any]:
    return json.loads(IG_SPEC_PATH.read_text())


def load_ig_cases(
    spec: dict[str, Any],
    dataset_set: str,
    source_path: pathlib.Path,
) -> list[MediQCase]:
    id_field = f"{dataset_set}_source_ids"
    source_ids = set(spec[id_field])
    rows = [json.loads(l) for l in source_path.read_text().strip().splitlines()]
    matched = [r for r in rows if r["id"] in source_ids]
    # Preserve order from spec
    ordered = sorted(matched, key=lambda r: spec[id_field].index(r["id"]))
    cases = []
    for r in ordered:
        ctx = r.get("context", [r.get("question", "")])
        facts = r.get("facts", ctx)
        sp = r.get("patient", {}).get("gpt_specialty", "Unknown")
        opts = {k: v for k, v in r.get("options", {}).items()}
        answer_idx = str(r.get("answer_idx", "A"))
        cases.append(
            MediQCase(
                case_id=f"mediq-{r['id']}",
                source_id=r["id"],
                specialty=sp,
                question=r["question"],
                initial_info=ctx[0] if ctx else r["question"],
                context=tuple(str(c) for c in ctx),
                facts=tuple(str(f) for f in facts),
                options=opts,
                answer_choice=answer_idx,
            )
        )
    return cases


def ig_dataset_fingerprint(cases: list[MediQCase], spec: dict[str, Any]) -> str:
    import hashlib
    fp_source = "|".join(str(c.source_id) for c in sorted(cases, key=lambda c: c.source_id))
    fp_source += f"|{spec['selection_fingerprint']}"
    return hashlib.sha256(fp_source.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_ig_metrics(results: list[dict]) -> dict[str, Any]:
    arms = list(ARMS)
    accuracy: dict[str, float] = {}
    n_correct: dict[str, int] = {}
    n_total = len(results)

    for arm in arms:
        correct = sum(
            1 for r in results
            if r["arms"].get(arm, {}).get("diagnosis", {}).get("correct", False)
        )
        n = sum(1 for r in results if arm in r.get("arms", {}))
        accuracy[arm] = correct / n if n > 0 else 0.0
        n_correct[arm] = correct

    paired: dict[str, dict] = {}
    for alt_arm in arms[1:]:
        wins = losses = ties = 0
        for r in results:
            a_correct = r["arms"].get("llm-freeform", {}).get("diagnosis", {}).get("correct", False)
            b_correct = r["arms"].get(alt_arm, {}).get("diagnosis", {}).get("correct", False)
            if b_correct and not a_correct:
                wins += 1
            elif a_correct and not b_correct:
                losses += 1
            else:
                ties += 1
        paired[f"{alt_arm}_vs_llm-freeform"] = {
            "ig_wins": wins,
            "baseline_wins": losses,
            "ties": ties,
            "mcnemar_exact_p": _mcnemar_exact_p(wins, losses),
        }

    ci: dict[str, tuple] = {
        arm: _wilson_interval(n_correct[arm], n_total) for arm in arms
    }

    mean_questions: dict[str, float] = {}
    mean_delta_h: dict[str, float] = {}
    for arm in arms:
        qs = [r["arms"][arm]["questions_asked"] for r in results if arm in r.get("arms", {})]
        mean_questions[arm] = statistics.mean(qs) if qs else 0.0
        deltas = [
            turn["selected_delta_h"]
            for r in results
            for turn in r["arms"].get(arm, {}).get("ig_trace", [])
        ]
        mean_delta_h[arm] = statistics.mean(deltas) if deltas else float("nan")

    return {
        "n_cases": n_total,
        "accuracy": {arm: round(accuracy[arm], 6) for arm in arms},
        "n_correct": n_correct,
        "accuracy_95ci": {arm: list(ci[arm]) for arm in arms},
        "paired": paired,
        "mean_questions": mean_questions,
        "mean_delta_h": mean_delta_h,
    }


# --------------------------------------------------------------------------
# JSONL logging
# --------------------------------------------------------------------------

def _log(path: pathlib.Path, event: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------

def run_ig_benchmark(
    cases: list[MediQCase],
    provider: Callable,
    retriever: TextbooksBM25Retriever,
    results_path: pathlib.Path,
    log_path: pathlib.Path,
    spec: dict[str, Any],
    dataset_set: str,
) -> dict[str, Any]:
    results: list[dict] = []
    total = len(cases)
    started = time.time()

    _log(log_path, {"event": "run_start", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "n_cases": total, "set": dataset_set})

    for pos, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            result = run_ig_case(case, provider, retriever)
            elapsed = time.perf_counter() - t0
            arm_correct = {
                arm: result["arms"][arm]["diagnosis"].get("correct", False)
                for arm in ARMS if arm in result["arms"]
            }
            status_str = result.get("status", "?")
            cost = provider.estimated_cost_usd if hasattr(provider, "estimated_cost_usd") else 0.0
            print(f"[{pos}/{total}] {case.case_id}: {status_str} "
                  f"ff={arm_correct.get('llm-freeform','?')} "
                  f"or={arm_correct.get('oracle-ig','?')} "
                  f"si={arm_correct.get('simulated-ig','?')} "
                  f"cost=${cost:.4f}")
            results.append(result)
            _log(log_path, {
                "event": "case_done", "position": pos, "case_id": case.case_id,
                "status": status_str, "arm_correct": arm_correct,
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[{pos}/{total}] {case.case_id}: FAILED {type(exc).__name__}: {exc}")
            _log(log_path, {
                "event": "case_failed", "position": pos, "case_id": case.case_id,
                "error_type": type(exc).__name__, "error": str(exc),
            })
            results.append({
                "case_id": case.case_id, "source_id": case.source_id,
                "specialty": case.specialty, "gold_choice": case.answer_choice,
                "status": "failed", "arms": {}, "model_trace": [],
                "error": str(exc),
            })

        # Save after every case
        metrics = compute_ig_metrics([r for r in results if r.get("status") != "failed"])
        output = {
            "experiment": "phase-ig",
            "set": dataset_set,
            "selection_fingerprint": spec["selection_fingerprint"],
            "results": results,
            "metrics": metrics,
        }
        results_path.write_text(json.dumps(output, indent=2, default=str))

    _log(log_path, {
        "event": "run_end",
        "cases_completed": len(results),
        "elapsed_s": round(time.time() - started, 1),
    })
    return output


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["dev", "holdout"], default="dev",
                        help="Which case set to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N cases (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--execute", action="store_true",
                        help="Required flag to actually run paid calls")
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("graphrag/eval/external/mediq/all_dev_good.jsonl"),
    )
    parser.add_argument("--index", type=pathlib.Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    spec = load_ig_spec()
    cases = load_ig_cases(spec, args.set, args.source)
    if args.limit:
        cases = cases[: args.limit]

    fingerprint = ig_dataset_fingerprint(cases, spec)

    if args.dry_run:
        calls_per_case = 3 * (N_CANDIDATES * 2 + 2) + 3  # rough
        est_cost = len(cases) * calls_per_case * 0.0015
        print(json.dumps({
            "set": args.set,
            "n_cases": len(cases),
            "dataset_fingerprint": fingerprint,
            "est_calls": len(cases) * calls_per_case,
            "est_cost_usd": round(est_cost, 2),
            "cost_guard_usd": spec[f"cost_guard_{args.set}_usd"],
        }, indent=2))
        return

    if not args.execute:
        print("Pass --execute to run paid calls (or --dry-run to preview).")
        return

    from dotenv import load_dotenv
    load_dotenv()

    checkpoint_path = DEFAULT_ROOT / f"ig_{args.set}_checkpoint.json"
    results_path = DEFAULT_ROOT / f"ig_{args.set}_results.json"
    log_path = DEFAULT_ROOT / f"ig_{args.set}_run_log.jsonl"

    provider = CheckpointedGeminiProvider(
        model="gemini-2.5-flash",
        checkpoint_path=checkpoint_path,
        fingerprint=fingerprint,
        max_calls=MAX_PROVIDER_CALLS,
        max_cost_usd=spec[f"cost_guard_{args.set}_usd"],
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    retriever = TextbooksBM25Retriever(args.index)

    output = run_ig_benchmark(cases, provider, retriever, results_path, log_path, spec, args.set)
    print(json.dumps({k: v for k, v in output["metrics"].items() if k != "paired"}, indent=2))
    print("Paired:", json.dumps(output["metrics"]["paired"], indent=2))


if __name__ == "__main__":
    main()
