import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from graphrag.eval.refusalbench_holdout import (
    ANSWER_BEHAVIOR,
    EXPECTED_ACTION,
    REFUSAL_BEHAVIORS,
    load_rows,
    prepare_holdout,
    select_health_holdout,
    verify_source,
)


def fixture_rows() -> list[dict]:
    rows = []
    for source_index in range(18):
        source_id = f"source-{source_index:02d}"
        behaviors = [ANSWER_BEHAVIOR, *REFUSAL_BEHAVIORS]
        for behavior_index, behavior in enumerate(behaviors):
            rows.append(
                {
                    "id": f"{source_id}-{behavior_index}",
                    "source_id": source_id,
                    "generator_model": "fixture",
                    "perturbation_class": f"P-{behavior_index}",
                    "intensity": "LOW" if behavior == ANSWER_BEHAVIOR else "HIGH",
                    "expected_rag_behavior": behavior,
                    "query": f"Question {source_index} {behavior_index}?",
                    "grounding": ["Evidence one", "Evidence two"],
                    "reference_answer": f"Answer {source_index}",
                    "verifier_votes": {"a": "PASS", "b": "PASS"},
                    "question_category": "Health",
                }
            )
    return rows


class RefusalBenchHoldoutTests(unittest.TestCase):
    def test_selection_is_source_paired_balanced_and_deterministic(self):
        first = select_health_holdout(fixture_rows(), seed=13)
        second = select_health_holdout(fixture_rows(), seed=13)
        self.assertEqual(first, second)
        self.assertEqual(first["case_count"], 36)
        self.assertEqual(first["source_pair_count"], 18)
        self.assertEqual(first["answerable_count"], 18)
        self.assertEqual(first["unanswerable_count"], 18)
        counts = Counter(case["expected_rag_behavior"] for case in first["cases"])
        self.assertEqual(counts[ANSWER_BEHAVIOR], 18)
        for behavior in REFUSAL_BEHAVIORS:
            self.assertEqual(counts[behavior], 3)
        pair_counts = Counter(case["pair_id"] for case in first["cases"])
        self.assertTrue(all(value == 2 for value in pair_counts.values()))
        self.assertEqual(len(first["selection_fingerprint"]), 64)

    def test_expected_actions_cover_all_behaviors(self):
        self.assertEqual(EXPECTED_ACTION[ANSWER_BEHAVIOR], "answer")
        self.assertEqual(EXPECTED_ACTION["REFUSE_AMBIGUOUS_QUERY"], "clarify")
        self.assertEqual(
            EXPECTED_ACTION["REFUSE_CONTRADICTORY_CONTEXT"], "escalate"
        )
        self.assertEqual(set(EXPECTED_ACTION), {ANSWER_BEHAVIOR, *REFUSAL_BEHAVIORS})

    def test_loader_normalizes_citation_text_and_rejects_duplicate_ids(self):
        row = fixture_rows()[0]
        row["grounding"] = [{"provider": "web", "cite_2": " second ", "cite_1": " first "}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_rows(path)
            self.assertEqual(loaded[0]["grounding"], ["first\nsecond"])
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_rows(path)

    def test_source_verification_and_prepare_are_zero_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in fixture_rows()),
                encoding="utf-8",
            )
            raw = source.read_bytes()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "repository": "fixture/repo",
                        "revision": "fixture-revision",
                        "file": source.name,
                        "url": "https://example.invalid/source.jsonl",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "license": "CC-BY-NC-4.0",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(verify_source(source, json.loads(manifest.read_text()))["valid"])
            result = prepare_holdout(
                source_path=source,
                output_path=None,
                manifest_path=manifest,
            )
        self.assertEqual(result["external_model_calls"], 0)
        self.assertEqual(result["estimated_api_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
