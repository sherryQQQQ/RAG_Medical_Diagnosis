"""Prepare a zero-call external Health holdout from RefusalBench-GaRAGe.

The raw CC-BY-NC-4.0 dataset stays under the gitignored external directory.
Selection is deterministic, source-paired, and balanced across the six refusal
categories: one answerable and one unanswerable variant for every Health source
question (18 source pairs / 36 cases).

This is an independently constructed, cross-model-verified RAG refusal holdout.
It is not clinician-reviewed and does not validate patient-specific medical use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import Request, urlopen


DEFAULT_MANIFEST = Path(__file__).parent / "specs" / "refusalbench_garage_manifest.json"
DEFAULT_ROOT = Path(__file__).parent / "external" / "refusalbench"
DEFAULT_SOURCE = DEFAULT_ROOT / "refusalbench-garage.jsonl"
DEFAULT_SELECTION = DEFAULT_ROOT / "stage5l_health_holdout.json"
SELECTION_SCHEMA_VERSION = 1
DEFAULT_SEED = 13
ANSWER_BEHAVIOR = "ANSWER_CORRECTLY"
REFUSAL_BEHAVIORS = (
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_NONFACTUAL_QUERY",
)
EXPECTED_ACTION = {
    ANSWER_BEHAVIOR: "answer",
    "REFUSE_AMBIGUOUS_QUERY": "clarify",
    "REFUSE_CONTRADICTORY_CONTEXT": "escalate",
    "REFUSE_FALSE_PREMISE_IN_QUERY": "abstain",
    "REFUSE_GRANULARITY_MISMATCH": "abstain",
    "REFUSE_INFO_MISSING_IN_CONTEXT": "abstain",
    "REFUSE_NONFACTUAL_QUERY": "abstain",
}
REQUIRED_FIELDS = {
    "id",
    "source_id",
    "generator_model",
    "perturbation_class",
    "intensity",
    "expected_rag_behavior",
    "query",
    "grounding",
    "reference_answer",
    "verifier_votes",
    "question_category",
}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"revision", "file", "url", "bytes", "sha256", "license"}
    if not required.issubset(payload):
        raise ValueError("RefusalBench manifest is incomplete")
    if payload["license"] != "CC-BY-NC-4.0":
        raise ValueError("Unexpected RefusalBench license")
    if len(str(payload["sha256"])) != 64 or int(payload["bytes"]) <= 0:
        raise ValueError("RefusalBench manifest has invalid source metadata")
    return payload


def verify_source(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"valid": False, "reason": "missing", "path": str(path)}
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    valid = size == int(manifest["bytes"]) and digest == manifest["sha256"]
    return {
        "valid": valid,
        "reason": "ok" if valid else "size_or_hash_mismatch",
        "path": str(path),
        "bytes": size,
        "sha256": digest,
    }


def download_source(
    path: Path = DEFAULT_SOURCE, manifest_path: Path = DEFAULT_MANIFEST
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    current = verify_source(path, manifest)
    if current["valid"]:
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        str(manifest["url"]),
        headers={"User-Agent": "medical-agent-answerability-eval/1.0"},
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, handle)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    checked = verify_source(temporary, manifest)
    if not checked["valid"]:
        temporary.unlink(missing_ok=True)
        raise ValueError("Downloaded RefusalBench source failed size/hash validation")
    temporary.replace(path)
    return verify_source(path, manifest)


def _grounding_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, Mapping):
        raise ValueError("Grounding passage must be a string or mapping")
    passage_parts = [
        str(value).strip()
        for key, value in sorted(item.items())
        if str(key).startswith("cite_") and str(value).strip()
    ]
    if not passage_parts:
        # Missing-information perturbations may intentionally blank a passage;
        # retain its position as an explicit absence signal for the evidence gate.
        return "[No textual evidence provided.]"
    return "\n".join(passage_parts)


def load_rows(path: Path = DEFAULT_SOURCE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or not REQUIRED_FIELDS.issubset(row):
                raise ValueError(f"Invalid RefusalBench row at line {line_number}")
            case_id = str(row["id"])
            if case_id in seen:
                raise ValueError(f"Duplicate RefusalBench ID: {case_id}")
            seen.add(case_id)
            if not isinstance(row["grounding"], list) or not row["grounding"]:
                raise ValueError(f"Case {case_id} has no grounding passages")
            normalized = dict(row)
            normalized["grounding"] = [
                _grounding_text(item) for item in row["grounding"]
            ]
            behavior = str(row["expected_rag_behavior"])
            if behavior not in EXPECTED_ACTION:
                raise ValueError(f"Unsupported expected behavior: {behavior}")
            rows.append(normalized)
    if not rows:
        raise ValueError("RefusalBench source is empty")
    return rows


def _stable_choice(rows: Iterable[dict[str, Any]], seed: int, salt: str) -> dict[str, Any]:
    candidates = list(rows)
    if not candidates:
        raise ValueError(f"No candidates for {salt}")
    rng = random.Random(f"{seed}:{salt}")
    return candidates[rng.randrange(len(candidates))]


def _assign_refusal_categories(
    by_source: Mapping[str, list[dict[str, Any]]], seed: int
) -> dict[str, str]:
    """Assign every Health source one refusal category, three sources each."""
    source_ids = sorted(by_source)
    if len(source_ids) != 18:
        raise ValueError(
            f"Pinned Health slice changed: expected 18 sources, found {len(source_ids)}"
        )
    target = {behavior: 3 for behavior in REFUSAL_BEHAVIORS}
    available = {
        source_id: {
            str(row["expected_rag_behavior"])
            for row in rows
            if row["expected_rag_behavior"] != ANSWER_BEHAVIOR
        }
        for source_id, rows in by_source.items()
    }
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    source_ids.sort(key=lambda source_id: len(available[source_id]))

    assignment: dict[str, str] = {}

    def visit(index: int) -> bool:
        if index == len(source_ids):
            return all(value == 0 for value in target.values())
        source_id = source_ids[index]
        choices = sorted(
            (behavior for behavior in available[source_id] if target.get(behavior, 0)),
            key=lambda behavior: (target[behavior], behavior),
        )
        rng.shuffle(choices)
        for behavior in choices:
            assignment[source_id] = behavior
            target[behavior] -= 1
            if visit(index + 1):
                return True
            target[behavior] += 1
            assignment.pop(source_id, None)
        return False

    if not visit(0):
        raise ValueError("Could not create balanced source-level refusal assignment")
    return assignment


def _case_payload(row: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    behavior = str(row["expected_rag_behavior"])
    return {
        "case_id": str(row["id"]),
        "pair_id": pair_id,
        "source_id": str(row["source_id"]),
        "query": str(row["query"]).strip(),
        "grounding": list(row["grounding"]),
        "reference_answer": str(row["reference_answer"]).strip(),
        "expected_rag_behavior": behavior,
        "expected_action": EXPECTED_ACTION[behavior],
        "answerable": behavior == ANSWER_BEHAVIOR,
        "perturbation_class": str(row["perturbation_class"]),
        "intensity": str(row["intensity"]),
        "generator_model": str(row["generator_model"]),
        "verifier_votes": dict(row["verifier_votes"]),
        "review_status": "cross_model_verified_not_clinician_reviewed",
    }


def select_health_holdout(
    rows: list[dict[str, Any]], seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    health = [row for row in rows if row["question_category"] == "Health"]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in health:
        by_source[str(row["source_id"])].append(row)
    assignment = _assign_refusal_categories(by_source, seed)

    cases: list[dict[str, Any]] = []
    for pair_number, source_id in enumerate(sorted(by_source), start=1):
        pair_id = f"health_pair_{pair_number:02d}"
        source_rows = by_source[source_id]
        answer = _stable_choice(
            (row for row in source_rows if row["expected_rag_behavior"] == ANSWER_BEHAVIOR),
            seed,
            f"{source_id}:answer",
        )
        refusal_behavior = assignment[source_id]
        refusal = _stable_choice(
            (
                row
                for row in source_rows
                if row["expected_rag_behavior"] == refusal_behavior
            ),
            seed,
            f"{source_id}:{refusal_behavior}",
        )
        cases.extend((_case_payload(answer, pair_id), _case_payload(refusal, pair_id)))

    behavior_counts = Counter(case["expected_rag_behavior"] for case in cases)
    if len(cases) != 36 or behavior_counts[ANSWER_BEHAVIOR] != 18:
        raise ValueError("Health holdout is not 18 balanced source pairs")
    if any(behavior_counts[behavior] != 3 for behavior in REFUSAL_BEHAVIORS):
        raise ValueError("Health holdout refusal categories are not balanced")
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Health holdout contains duplicate cases")

    selection_fingerprint = fingerprint(cases)
    return {
        "evaluation_type": "external_refusalbench_health_holdout",
        "schema_version": SELECTION_SCHEMA_VERSION,
        "seed": seed,
        "selection_fingerprint": selection_fingerprint,
        "case_count": len(cases),
        "source_pair_count": len(by_source),
        "answerable_count": behavior_counts[ANSWER_BEHAVIOR],
        "unanswerable_count": len(cases) - behavior_counts[ANSWER_BEHAVIOR],
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "scope": "external_cross_model_verified_not_clinician_reviewed",
        "license": "CC-BY-NC-4.0",
        "external_model_calls": 0,
        "estimated_api_cost_usd": 0.0,
        "cases": cases,
    }


def prepare_holdout(
    *,
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path | None = DEFAULT_SELECTION,
    manifest_path: Path = DEFAULT_MANIFEST,
    seed: int = DEFAULT_SEED,
    download: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if download:
        source_status = download_source(source_path, manifest_path)
    else:
        source_status = verify_source(source_path, manifest)
        if not source_status["valid"]:
            raise FileNotFoundError(
                f"Pinned RefusalBench source is unavailable or invalid: {source_status}"
            )
    selection = select_health_holdout(load_rows(source_path), seed)
    selection["source"] = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "file": manifest["file"],
        "sha256": manifest["sha256"],
        "manifest_fingerprint": fingerprint(manifest),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selection = prepare_holdout(
        source_path=args.source,
        output_path=None if args.dry_run else args.output,
        manifest_path=args.manifest,
        seed=args.seed,
        download=args.download,
    )
    print("Stage 5L external RefusalBench Health holdout")
    print(
        f"Cases: {selection['case_count']} "
        f"({selection['source_pair_count']} source pairs; "
        f"{selection['answerable_count']} answerable / "
        f"{selection['unanswerable_count']} unanswerable)"
    )
    print(f"Selection fingerprint: {selection['selection_fingerprint']}")
    print(f"Behavior counts: {selection['behavior_counts']}")
    print("External model calls: 0; estimated API cost: US$0.00")
    if not args.dry_run:
        print(f"Local gitignored selection: {args.output}")


if __name__ == "__main__":
    main()
