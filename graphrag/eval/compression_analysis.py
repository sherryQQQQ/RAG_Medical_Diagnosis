"""Phase 0 zero-cost compression analysis over Stage 5N/5O checkpoints.

Reads the committed MediQ handoff result JSONs (no model calls) and produces:

1. Compression accounting (P0.1): per-condition diagnostic-call input tokens
   (exact, from provider usage metadata) and representation-level sizes
   (heuristic, reconstructed exactly as ``_diagnosis_prompt`` serialises the
   patient input; token estimate = chars / 4, labelled heuristic).
2. Fact-preservation audit (P0.2): lexical key-fact recall/precision
   (existing per-case metrics aggregated), negation retention (heuristic
   lexical cues), and fabricated-fact rate (facts citing nonexistent patient
   turns; fail-closed design predicts zero).
3. Failure catalog (P0.3): every 5O case wrong in at least one condition,
   labelled with a root-cause category derived from the cross-condition
   correctness pattern and per-case handoff fact recall.

Usage:
    python -m graphrag.eval.compression_analysis

Writes graphrag/eval/data/compression_analysis_metrics.json and prints a
summary. The Markdown report in docs/compression_analysis.md is written from
this JSON.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MEDIQ_DIR = ROOT / "graphrag" / "eval" / "external" / "mediq"
OUT_PATH = ROOT / "graphrag" / "eval" / "data" / "compression_analysis_metrics.json"

NEGATION_CUES = re.compile(
    r"\b(no|not|denies|denied|without|absent|negative|never|unremarkable|non[- ]?tender|afebrile)\b",
    re.IGNORECASE,
)

STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his in is it its of on or "
    "she that the their there they this to was were with".split()
)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _lexical_match(source: str, targets: list[str], threshold: float = 0.5) -> bool:
    src = _content_tokens(source)
    if not src:
        return False
    return any(len(src & _content_tokens(t)) / len(src) >= threshold for t in targets)


def _est_tokens(chars: int) -> float:
    """Heuristic character-based token estimate (chars / 4); not provider truth."""
    return chars / 4.0


def _representation_sizes(case: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the patient-input serialisation used by _diagnosis_prompt."""
    conversation = case.get("conversation") or []
    handoff = case.get("handoff") or {}
    transcript_repr = json.dumps(conversation, indent=2)
    handoff_repr = json.dumps(handoff, indent=2)
    cited_turn_ids = {
        turn_id
        for fact in handoff.get("facts", [])
        for turn_id in fact.get("source_turn_ids", [])
    }
    cited_turns = [t for t in conversation if t.get("turn_id") in cited_turn_ids]
    handoff_plus_repr = (
        handoff_repr + "\n\nCited raw patient turns:\n" + json.dumps(cited_turns, indent=2)
    )
    sizes = {
        "full-transcript": len(transcript_repr),
        "structured-handoff": len(handoff_repr),
        "handoff-plus-sources": len(handoff_plus_repr),
    }
    ratio = (
        sizes["structured-handoff"] / sizes["full-transcript"]
        if sizes["full-transcript"]
        else None
    )
    return {
        "chars": sizes,
        "est_tokens": {k: round(_est_tokens(v), 1) for k, v in sizes.items()},
        "handoff_over_transcript_ratio": round(ratio, 4) if ratio else None,
    }


def _diag_tokens(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cond, diag in (case.get("diagnoses") or {}).items():
        out[cond] = {
            "input_tokens": diag.get("input_tokens"),
            "output_tokens": diag.get("output_tokens"),
            "correct": diag.get("correct"),
            "status": diag.get("status"),
            "error": diag.get("error", ""),
        }
    return out


def _negation_audit(case: dict[str, Any]) -> dict[str, Any]:
    revealed = case.get("revealed_patient_facts") or {}
    handoff = case.get("handoff") or {}
    fact_statements = [f.get("statement", "") for f in handoff.get("facts", [])]
    negated_sources = {
        fid: text for fid, text in revealed.items() if NEGATION_CUES.search(text)
    }
    retained = {
        fid: _lexical_match(text, fact_statements)
        for fid, text in negated_sources.items()
    }
    status_counts: dict[str, int] = {}
    for fact in handoff.get("facts", []):
        status = fact.get("status", "present")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "negated_source_facts": len(negated_sources),
        "negated_source_facts_retained": sum(retained.values()),
        "handoff_status_counts": status_counts,
    }


def _fabricated_facts(case: dict[str, Any]) -> list[str]:
    patient_turns = {
        t.get("turn_id")
        for t in case.get("conversation") or []
        if t.get("role") == "patient"
    }
    bad: list[str] = []
    for fact in (case.get("handoff") or {}).get("facts", []):
        cited = fact.get("source_turn_ids", [])
        if not cited or any(tid not in patient_turns for tid in cited):
            bad.append(fact.get("fact_id", "?"))
    return bad


def _classify_failure(
    per_cond: dict[str, dict[str, Any]], fact_recall: float | None
) -> str:
    """Heuristic root-cause label from the cross-condition correctness pattern.

    These labels are pattern-based inferences, not causal proof.
    """
    ft = per_cond.get("full-transcript", {}).get("correct")
    sh = per_cond.get("structured-handoff", {}).get("correct")
    hp = per_cond.get("handoff-plus-sources", {}).get("correct")
    if ft is False and sh is False and hp is False:
        return "information-acquisition-or-intrinsic (wrong in all conditions)"
    if ft is True and sh is False:
        if fact_recall is not None and fact_recall < 0.75:
            return "handoff-loss-candidate (transcript right, handoff wrong, low fact recall)"
        return "generation-variance-candidate (transcript right, handoff wrong, high fact recall)"
    if ft is False and (sh is True or hp is True):
        return "transcript-noise-candidate (handoff right, transcript wrong)"
    return "mixed (see per-condition detail)"


def analyse_stage(path: Path, transcript_condition: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    cases = data["results"]
    conditions = sorted({c for case in cases for c in (case.get("diagnoses") or {})})

    per_case: list[dict[str, Any]] = []
    for case in cases:
        diag = _diag_tokens(case)
        repr_sizes = _representation_sizes(case)
        negation = _negation_audit(case)
        fabricated = _fabricated_facts(case)
        hm = case.get("handoff_metrics") or {}
        per_case.append(
            {
                "case_id": case.get("case_id"),
                "specialty": case.get("specialty"),
                "questions_asked": case.get("questions_asked"),
                "gold_choice": case.get("gold_choice"),
                "diagnostic_calls": diag,
                "representation": repr_sizes,
                "fact_recall": hm.get("fact_recall"),
                "fact_precision": hm.get("fact_precision"),
                "negation": negation,
                "fabricated_fact_ids": fabricated,
            }
        )

    def _cond_stats(key: str) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for cond in conditions:
            values = [
                c["diagnostic_calls"][cond][key]
                for c in per_case
                if c["diagnostic_calls"].get(cond, {}).get(key) is not None
            ]
            if values:
                stats[cond] = {
                    "mean": round(statistics.mean(values), 1),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }
        return stats

    ratios = [
        c["representation"]["handoff_over_transcript_ratio"]
        for c in per_case
        if c["representation"]["handoff_over_transcript_ratio"] is not None
    ]
    input_token_stats = _cond_stats("input_tokens")
    tokens_vs_transcript = {}
    base = input_token_stats.get(transcript_condition, {}).get("mean")
    if base:
        tokens_vs_transcript = {
            cond: round(stats["mean"] / base, 4)
            for cond, stats in input_token_stats.items()
        }

    recalls = [c["fact_recall"] for c in per_case if c["fact_recall"] is not None]
    precisions = [c["fact_precision"] for c in per_case if c["fact_precision"] is not None]
    neg_total = sum(c["negation"]["negated_source_facts"] for c in per_case)
    neg_kept = sum(c["negation"]["negated_source_facts_retained"] for c in per_case)
    fabricated_total = sum(len(c["fabricated_fact_ids"]) for c in per_case)
    total_facts = sum(
        sum(c["negation"]["handoff_status_counts"].values()) for c in per_case
    )

    failures = []
    for c in per_case:
        wrong_any = any(
            d.get("correct") is False for d in c["diagnostic_calls"].values()
        )
        if wrong_any:
            failures.append(
                {
                    "case_id": c["case_id"],
                    "specialty": c["specialty"],
                    "questions_asked": c["questions_asked"],
                    "correct_by_condition": {
                        cond: d.get("correct")
                        for cond, d in c["diagnostic_calls"].items()
                    },
                    "fact_recall": c["fact_recall"],
                    "root_cause_label": _classify_failure(
                        c["diagnostic_calls"], c["fact_recall"]
                    ),
                }
            )
    label_counts: dict[str, int] = {}
    for f in failures:
        label_counts[f["root_cause_label"]] = label_counts.get(f["root_cause_label"], 0) + 1

    return {
        "source_file": str(path.relative_to(ROOT)),
        "stage": data.get("stage"),
        "model": data.get("model"),
        "n_cases": len(cases),
        "transcript_condition": transcript_condition,
        "compression": {
            "diagnostic_input_tokens_exact": input_token_stats,
            "diagnostic_input_tokens_vs_transcript_mean_ratio": tokens_vs_transcript,
            "representation_ratio_handoff_over_transcript": {
                "mean": round(statistics.mean(ratios), 4) if ratios else None,
                "median": round(statistics.median(ratios), 4) if ratios else None,
                "min": round(min(ratios), 4) if ratios else None,
                "max": round(max(ratios), 4) if ratios else None,
                "note": "char-based heuristic on the serialised patient input only",
            },
        },
        "fact_preservation": {
            "mean_lexical_fact_recall": round(statistics.mean(recalls), 4) if recalls else None,
            "mean_lexical_fact_precision": round(statistics.mean(precisions), 4)
            if precisions
            else None,
            "negated_source_facts": neg_total,
            "negated_source_facts_retained": neg_kept,
            "negation_retention_rate": round(neg_kept / neg_total, 4) if neg_total else None,
            "total_handoff_facts": total_facts,
            "fabricated_fact_count": fabricated_total,
            "fabricated_fact_rate": round(fabricated_total / total_facts, 4)
            if total_facts
            else None,
            "note": "negation retention and fabrication checks are lexical heuristics",
        },
        "failure_catalog": {
            "cases_wrong_in_at_least_one_condition": len(failures),
            "root_cause_label_counts": label_counts,
            "cases": failures,
        },
        "per_case": per_case,
    }


def main() -> None:
    report = {
        "format_version": 1,
        "phase": "P0 zero-cost compression analysis",
        "model_calls_made": 0,
        "stages": {
            "5O": analyse_stage(
                MEDIQ_DIR / "stage5o_handoff_results.json", "full-transcript"
            ),
            "5N": analyse_stage(
                MEDIQ_DIR / "stage5n_handoff_results.json", "raw-conversation"
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=1))
    for stage_name, stage in report["stages"].items():
        print(f"== Stage {stage_name} ({stage['n_cases']} cases)")
        print(json.dumps(stage["compression"], indent=1))
        print(json.dumps(stage["fact_preservation"], indent=1))
        print(
            "failure labels:",
            json.dumps(stage["failure_catalog"]["root_cause_label_counts"], indent=1),
        )
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
