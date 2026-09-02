"""
Phase 2 — Distractor Injection Stress Test.

Injects k ∈ {0, 10, 25} diagnostically-irrelevant Q&A turns into each
transcript, rebuilds the handoff from the padded transcript, then runs three
diagnostic arms (full-transcript, freetext-summary, structured-handoff).

The 30-case subset and distractor pool are frozen in:
  graphrag/eval/specs/mediq_phase2_spec.json
  graphrag/eval/specs/phase2_distractors.json

Phase 1 interview calls are checkpoint-reused (free). New calls for k > 0:
  - 1 handoff-rebuild call per case × k-level
  - 1 diagnose call × arm × k-level
  - 1 summarize call for freetext-summary × k-level

Usage:
    python -m graphrag.eval.mediq_phase2 [--dry-run] [--execute] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import time
from dataclasses import asdict
from typing import Any, Callable, Sequence

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
    _interview_prompt,
    _summary_prompt,
    INTERVIEW_SCHEMA,
    SUMMARY_SCHEMA,
    diagnosis_schema,
    parse_diagnosis,
    parse_interview_update,
)
from graphrag.eval.mediq_handoff_data import (
    DEFAULT_ROOT,
    MediQCase,
)
from graphrag.eval.mirage_benchmark import (
    _mcnemar_exact_p,
    _wilson_interval,
)
from graphrag.eval.mirage_corpus import DEFAULT_INDEX, TextbooksBM25Retriever

SPEC_PATH = pathlib.Path("graphrag/eval/specs/mediq_phase2_spec.json")
DISTRACTOR_PATH = pathlib.Path("graphrag/eval/specs/phase2_distractors.json")
P1_RESULTS_PATH = DEFAULT_ROOT / "p1_compaction_results.json"
P1_CHECKPOINT = DEFAULT_ROOT / "p1_provider_checkpoint.json"

P2_CHECKPOINT = DEFAULT_ROOT / "p2_provider_checkpoint.json"
P2_RESULTS = DEFAULT_ROOT / "p2_results.json"
P2_LOG = DEFAULT_ROOT / "p2_run_log.jsonl"

K_LEVELS = [0, 10, 25]
ARMS = ["full-transcript", "freetext-summary", "structured-handoff"]
MAX_OUTPUT_TOKENS = 2_048
MAX_PROMPT_CHARS = 16_000
MAX_PROVIDER_CALLS = 3_000


# --------------------------------------------------------------------------
# Distractor injection
# --------------------------------------------------------------------------

def _inject_distractors(
    conversation: list[dict],
    distractors: list[dict],
    k: int,
    case_id: str,
    base_turn_offset: int,
) -> list[dict]:
    """Return a copy of conversation with k distractors inserted at random
    positions (excluding turn-0). Positions are deterministic (seed from
    case_id + k so every arm gets the same padded transcript)."""
    if k == 0:
        return list(conversation)
    rng = random.Random(f"{case_id}:{k}")
    chosen = rng.sample(distractors[:100], min(k, len(distractors)))

    # Positions to insert: after turn 0 (initial info), before the end
    n = len(conversation)
    positions = sorted(rng.sample(range(1, n + 1), min(k, n)))

    padded = list(conversation)
    offset = base_turn_offset
    for i, (pos, d) in enumerate(zip(positions, chosen)):
        insert_at = pos + i * 2  # each insertion shifts subsequent positions
        agent_id = f"distractor-agent-{offset + i}"
        patient_id = f"distractor-patient-{offset + i}"
        padded.insert(insert_at, {"turn_id": agent_id, "role": "agent", "content": d["question"]})
        padded.insert(insert_at + 1, {"turn_id": patient_id, "role": "patient", "content": d["answer"]})
    return padded


# --------------------------------------------------------------------------
# Handoff rebuild from padded transcript
# --------------------------------------------------------------------------

def _rebuild_handoff(
    case: MediQCase,
    padded_conv: list[dict],
    provider: Callable,
    record: Callable,
    k: int,
) -> ClinicalHandoff | None:
    """Run one interview call on the padded transcript to get updated handoff."""
    conversation = tuple(
        ConversationTurn(t["turn_id"], t["role"], t["content"]) for t in padded_conv
    )
    call_id = f"{case.case_id}:p2:rebuild:{k}"
    prompt = _interview_prompt(case, conversation, None, max_questions=len(padded_conv))
    # Prompt it to finalize: we're not asking another question, just summarizing
    prompt = prompt.replace(
        "either ask one targeted question or finalize",
        "finalize the handoff — do not ask another question",
    )

    last_error = None
    for attempt, (rid, rlabel) in enumerate([
        (None, f"p2-rebuild-k{k}"),
        ("retry-json-v1", f"p2-rebuild-k{k}-retry-json"),
        ("retry-contract-v1", f"p2-rebuild-k{k}-retry-contract"),
    ]):
        if attempt == 0:
            resp = provider(call_id, rlabel, prompt, INTERVIEW_SCHEMA)
        else:
            resp = provider(f"{call_id}:{rid}", rlabel, prompt, INTERVIEW_SCHEMA)
        record(rlabel, resp)
        try:
            _json_object(resp.raw_output)
            update = parse_interview_update(resp.raw_output, conversation)
            return update.handoff
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    return None  # all retries failed


# --------------------------------------------------------------------------
# Diagnosis for one arm
# --------------------------------------------------------------------------

def _diagnose_arm(
    case: MediQCase,
    arm: str,
    handoff: ClinicalHandoff | None,
    padded_conv: list[dict],
    evidence: tuple[EvidenceItem, ...],
    provider: Callable,
    record: Callable,
    k: int,
) -> dict[str, Any]:
    if handoff is None and arm == "structured-handoff":
        return {"answer_choice": None, "correct": False, "status": "no_handoff",
                "input_tokens": 0, "output_tokens": 0, "confidence": 0.0, "error": ""}

    conversation = tuple(
        ConversationTurn(t["turn_id"], t["role"], t["content"]) for t in padded_conv
    )
    summary_text = ""

    if arm == "freetext-summary":
        sum_call_id = f"{case.case_id}:p2:summarize:k{k}"
        sum_prompt = _summary_prompt(case, conversation)
        sum_resp = provider(sum_call_id, f"p2-summarize-k{k}", sum_prompt, SUMMARY_SCHEMA)
        record(f"p2-summarize-k{k}", sum_resp)
        try:
            summary_text = str(_json_object(sum_resp.raw_output).get("summary", "")).strip()
        except Exception:
            summary_text = ""

    packet = DiagnosticPacket(
        handoff=handoff or ClinicalHandoff(
            chief_complaint=case.initial_info, facts=()
        ),
        evidence=evidence,
        source_turns=tuple(t for t in conversation if t.role == "patient"),
    )
    prompt, allowed_ids = _diagnosis_prompt(
        case, packet, arm,
        conversation=conversation,
        summary_text=summary_text,
    )
    call_id = f"{case.case_id}:p2:diagnose:{arm}:k{k}"

    parsed = None
    final_error = None
    last_resp = None
    for retry_n in range(3):
        suffix = "" if retry_n == 0 else f":retry-v{retry_n}"
        stage = f"p2-diagnose-{arm}-k{k}{suffix}"
        resp = provider(call_id + suffix, stage, prompt, diagnosis_schema(case.options))
        record(stage, resp)
        last_resp = resp
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

    if final_error or parsed is None:
        return {
            "answer_choice": None, "correct": False, "status": "error",
            "error": str(final_error),
            "input_tokens": last_resp.input_tokens if last_resp else 0,
            "output_tokens": last_resp.output_tokens if last_resp else 0,
            "confidence": 0.0,
        }
    return {
        **parsed,
        "correct": parsed["answer_choice"] == case.answer_choice,
        "status": "completed",
        "error": "",
        "input_tokens": last_resp.input_tokens if last_resp else 0,
        "output_tokens": last_resp.output_tokens if last_resp else 0,
    }


# --------------------------------------------------------------------------
# Per-case runner
# --------------------------------------------------------------------------

def run_phase2_case(
    case: MediQCase,
    p1_result: dict,
    distractors: list[dict],
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

    original_conv = p1_result.get("conversation", [])
    base_offset = len(original_conv)

    # Retrieve evidence (once per case, same handoff-based query as P1)
    p1_handoff_raw = p1_result.get("handoff")
    if p1_handoff_raw:
        from graphrag.agent.clinical_handoff import ClinicalFact
        facts_text = " ".join(
            f.get("statement", "") for f in p1_handoff_raw.get("facts", [])
        )
        query = f"{case.question} {facts_text}"
    else:
        query = case.question
    snippets = retriever.retrieve(query, k=8)
    evidence = tuple(
        EvidenceItem(s.snippet_id, s.content, s.title) for s in snippets
    )

    results_by_k: dict[int, dict] = {}

    for k in K_LEVELS:
        padded_conv = _inject_distractors(
            original_conv, distractors, k, case.case_id, base_offset
        )

        if k == 0:
            # Reuse P1 handoff directly (checkpoint_reused = True conceptually)
            handoff = _p1_handoff(p1_result)
        else:
            handoff = _rebuild_handoff(case, padded_conv, provider, record, k)

        arm_results: dict[str, dict] = {}
        for arm in ARMS:
            arm_results[arm] = _diagnose_arm(
                case, arm, handoff, padded_conv, evidence, provider, record, k
            )

        n_distractor_patient = sum(
            1 for t in padded_conv
            if t["role"] == "patient" and t["turn_id"].startswith("distractor")
        )
        results_by_k[k] = {
            "k": k,
            "n_padded_turns": len(padded_conv),
            "n_distractor_turns": n_distractor_patient,
            "handoff_rebuilt": k > 0,
            "arms": arm_results,
        }

    return {
        "case_id": case.case_id,
        "source_id": case.source_id,
        "specialty": case.specialty,
        "gold_choice": case.answer_choice,
        "k_results": results_by_k,
        "model_trace": model_trace,
    }


def _p1_handoff(p1_result: dict) -> ClinicalHandoff | None:
    """Reconstruct ClinicalHandoff from stored P1 result dict."""
    raw = p1_result.get("handoff")
    if not raw:
        return None
    try:
        from graphrag.agent.clinical_handoff import ClinicalFact
        facts = tuple(
            ClinicalFact(
                fact_id=str(f["fact_id"]),
                category=str(f["category"]),
                statement=str(f["statement"]),
                status=str(f["status"]),
                source_turn_ids=tuple(str(s) for s in f["source_turn_ids"]),
            )
            for f in raw.get("facts", [])
            if f.get("source_turn_ids")
        )
        return ClinicalHandoff(
            chief_complaint=str(raw.get("chief_complaint", "")),
            facts=facts,
            missing_information=tuple(str(m) for m in raw.get("missing_information", [])),
            contradictions=tuple(str(c) for c in raw.get("contradictions", [])),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_phase2_metrics(results: list[dict]) -> dict[str, Any]:
    metrics: dict[int, dict] = {}
    for k in K_LEVELS:
        accuracy: dict[str, float] = {}
        n_correct: dict[str, int] = {}
        n_valid = 0
        for r in results:
            kres = r.get("k_results", {}).get(k, {})
            if not kres:
                continue
            n_valid += 1
            for arm in ARMS:
                c = kres.get("arms", {}).get(arm, {}).get("correct", False)
                n_correct[arm] = n_correct.get(arm, 0) + int(c)

        for arm in ARMS:
            accuracy[arm] = n_correct.get(arm, 0) / n_valid if n_valid > 0 else 0.0

        # Paired McNemar: structured-handoff vs full-transcript
        wins = losses = ties = 0
        for r in results:
            kres = r.get("k_results", {}).get(k, {})
            if not kres:
                continue
            sh = kres["arms"].get("structured-handoff", {}).get("correct", False)
            ft = kres["arms"].get("full-transcript", {}).get("correct", False)
            if sh and not ft:
                wins += 1
            elif ft and not sh:
                losses += 1
            else:
                ties += 1

        ci = {arm: list(_wilson_interval(n_correct.get(arm, 0), n_valid)) for arm in ARMS}

        # Mean input tokens per arm per k
        mean_tokens: dict[str, float] = {}
        for arm in ARMS:
            toks = [
                r["k_results"][k]["arms"][arm]["input_tokens"]
                for r in results if k in r.get("k_results", {})
                and arm in r["k_results"][k].get("arms", {})
            ]
            mean_tokens[arm] = statistics.mean(toks) if toks else 0.0

        metrics[k] = {
            "k": k,
            "n_valid": n_valid,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "accuracy_95ci": ci,
            "mcnemar_sh_vs_ft": {
                "sh_wins": wins,
                "ft_wins": losses,
                "ties": ties,
                "p": _mcnemar_exact_p(wins, losses),
            },
            "mean_input_tokens": mean_tokens,
        }
    return metrics


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
# Dataset loading
# --------------------------------------------------------------------------

def load_phase2_cases(spec: dict, p1_results: dict) -> list[tuple[MediQCase, dict]]:
    """Returns list of (MediQCase, p1_result) pairs for the 30-case subset."""
    selected_ids = set(spec["source_cases"])
    p1_by_id = {r["case_id"]: r for r in p1_results["results"]}

    # Load full case metadata from MediQ source
    source_path = pathlib.Path("graphrag/eval/external/mediq/all_dev_good.jsonl")
    rows_by_id = {}
    for line in source_path.read_text().strip().splitlines():
        r = json.loads(line)
        cid = f"mediq-{r['id']}"
        if cid in selected_ids:
            rows_by_id[cid] = r

    pairs: list[tuple[MediQCase, dict]] = []
    for cid in spec["source_cases"]:
        if cid not in rows_by_id or cid not in p1_by_id:
            continue
        r = rows_by_id[cid]
        ctx = r.get("context", [r.get("question", "")])
        facts = r.get("facts", ctx)
        opts = {k: v for k, v in r.get("options", {}).items()}
        answer_idx = str(r.get("answer_idx", "A"))
        sp = r.get("patient", {}).get("gpt_specialty", "Unknown")
        case = MediQCase(
            case_id=cid,
            source_id=r["id"],
            specialty=sp,
            question=r["question"],
            initial_info=ctx[0] if ctx else r["question"],
            context=tuple(str(c) for c in ctx),
            facts=tuple(str(f) for f in facts),
            options=opts,
            answer_choice=answer_idx,
        )
        pairs.append((case, p1_by_id[cid]))
    return pairs


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_phase2(
    pairs: list[tuple[MediQCase, dict]],
    distractors: list[dict],
    provider: Callable,
    retriever: TextbooksBM25Retriever,
    results_path: pathlib.Path,
    log_path: pathlib.Path,
    spec: dict,
) -> dict[str, Any]:
    results: list[dict] = []
    total = len(pairs)
    started = time.time()

    _log(log_path, {"event": "run_start", "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "n_cases": total, "k_levels": K_LEVELS, "arms": ARMS})

    for pos, (case, p1_res) in enumerate(pairs, 1):
        t0 = time.perf_counter()
        try:
            result = run_phase2_case(case, p1_res, distractors, provider, retriever)
            elapsed = time.perf_counter() - t0
            cost = getattr(provider, "estimated_cost_usd", 0.0)
            summary = {
                k: {arm: result["k_results"][k]["arms"][arm].get("correct", "?")
                    for arm in ARMS}
                for k in K_LEVELS if k in result["k_results"]
            }
            print(f"[{pos}/{total}] {case.case_id}: cost=${cost:.4f} {summary}")
            results.append(result)
            _log(log_path, {
                "event": "case_done", "position": pos, "case_id": case.case_id,
                "elapsed_s": round(elapsed, 1), "k_summary": summary,
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
                "status": "failed", "k_results": {}, "model_trace": [],
            })

        metrics = compute_phase2_metrics(results)
        output = {
            "experiment": "phase-2",
            "selection_fingerprint": spec["selection_fingerprint"],
            "distractor_fingerprint": spec["distractor_fingerprint"],
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", type=pathlib.Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text())
    distractor_data = json.loads(DISTRACTOR_PATH.read_text())
    distractors = distractor_data["distractors"]
    p1_results = json.loads(P1_RESULTS_PATH.read_text())

    pairs = load_phase2_cases(spec, p1_results)
    if args.limit:
        pairs = pairs[: args.limit]

    if args.dry_run:
        # k=0 costs nothing (P1 handoffs reused). k=10 and k=25 each need:
        # 1 rebuild + 3 diagnose + 1 summarize = 5 calls per case
        new_calls = len(pairs) * 2 * 5
        est_cost = new_calls * 0.001
        print(json.dumps({
            "n_cases": len(pairs),
            "k_levels": K_LEVELS,
            "arms": ARMS,
            "selection_fingerprint": spec["selection_fingerprint"],
            "distractor_fingerprint": spec["distractor_fingerprint"],
            "new_calls_est": new_calls,
            "est_cost_usd": round(est_cost, 2),
            "cost_guard_usd": spec["cost_guard_usd"],
        }, indent=2))
        return

    if not args.execute:
        print("Pass --execute to run (or --dry-run to preview).")
        return

    from dotenv import load_dotenv
    load_dotenv()

    import hashlib
    fp_src = "|".join(cid for case, _ in pairs for cid in [case.case_id])
    fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()[:16]

    provider = CheckpointedGeminiProvider(
        model="gemini-2.5-flash",
        checkpoint_path=P2_CHECKPOINT,
        fingerprint=fingerprint,
        max_calls=MAX_PROVIDER_CALLS,
        max_cost_usd=spec["cost_guard_usd"],
        max_prompt_chars=MAX_PROMPT_CHARS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    retriever = TextbooksBM25Retriever(args.index)

    output = run_phase2(pairs, distractors, provider, retriever, P2_RESULTS, P2_LOG, spec)
    print(json.dumps(output["metrics"], indent=2, default=str))


if __name__ == "__main__":
    main()
