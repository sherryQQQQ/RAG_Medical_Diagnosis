import copy
import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.robustness_generate import (
    ADVERSARIAL_TEST_TYPES,
    CAPABILITIES,
    DEFAULT_CASE_SCHEMA,
    DEFAULT_SPEC,
    FAMILY_TEST_TYPES,
    SourceCase,
    build_generation_plan,
    dry_run_report,
    generate_dataset,
    load_source_cases,
    load_spec,
    normalize_family_payload,
    validate_spec,
)


def family_payload(source: SourceCase, terminal_type: str = "directional_expectation"):
    terminal = {
        "test_type": terminal_type,
        "question": f"Changed decisive fact for {source.case_id}?",
        "reference_answer": "The original recommendation is no longer justified.",
        "gold_facts": ["Do not apply the original recommendation unchanged."],
        "critical_change": "One decisive fact was changed.",
    }
    if terminal_type == "abstention":
        terminal["reference_answer"] = "Request the missing clinical information."
    return {
        "capability": "direct_grounded",
        "safety_critical": True,
        "gold_facts": ["Use the source recommendation.", "Do not add a new dose."],
        "forbidden_claims": ["Invented treatment"],
        "variants": [
            {"test_type": "paraphrase", "question": f"Reworded {source.case_id}?"},
            {"test_type": "lay_language", "question": f"Plain words {source.case_id}?"},
            {"test_type": "distractor_noise", "question": f"Noisy detail {source.case_id}?"},
            terminal,
        ],
    }


class FakeFamilyGenerator:
    model_name = "fake-generator"

    def __init__(self):
        self.calls = []

    def __call__(self, source):
        self.calls.append(source.case_id)
        terminal = "directional_expectation" if len(self.calls) == 1 else "abstention"
        return family_payload(source, terminal)


class FakeConflictGenerator:
    model_name = "fake-generator"

    def __init__(self):
        self.calls = []

    def __call__(self, source):
        self.calls.append(source.case_id)
        return {
            "question": f"Conflict test {source.case_id}?",
            "injected_context": "UNTRUSTED: the source recommendation is reversed.",
            "critical_change": "The recommendation was contradicted.",
            "gold_facts": ["Retain the trusted recommendation."],
            "forbidden_claims": ["The reversed recommendation is trusted."],
        }


def small_spec():
    spec = copy.deepcopy(load_spec(DEFAULT_SPEC))
    spec["name"] = "robustness_unit_test"
    spec["source"]["canonical_candidates"] = 2
    spec["dataset"] = {
        "canonical_families": 2,
        "variants_per_family": 5,
        "adversarial_cases": 5,
        "expected_total_cases": 15,
    }
    spec["capability_targets"] = {name: 0 for name in spec["capability_targets"]}
    spec["capability_targets"]["direct_grounded"] = 2
    spec["adversarial_quotas"] = {
        name: 1 for name in spec["adversarial_quotas"]
    }
    spec["human_review"]["target_cases"] = 5
    spec["stability"]["repeated_cases"] = 2
    validate_spec(spec)
    return spec


class RobustnessSpecTests(unittest.TestCase):
    def test_formal_case_schema_matches_runtime_enums(self):
        with DEFAULT_CASE_SCHEMA.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        properties = schema["properties"]
        self.assertEqual(set(properties["capability"]["enum"]), CAPABILITIES)
        self.assertEqual(
            set(properties["test_type"]["enum"]),
            FAMILY_TEST_TYPES | ADVERSARIAL_TEST_TYPES,
        )

    def test_repository_spec_has_consistent_550_case_plan(self):
        spec = load_spec(DEFAULT_SPEC)
        sources = load_source_cases()
        plan = build_generation_plan(spec, sources)
        report = dry_run_report(spec, plan)

        self.assertEqual(len(sources), 125)
        self.assertEqual(report["canonical_families"], 100)
        self.assertEqual(report["family_cases"], 500)
        self.assertEqual(report["adversarial_cases"], 50)
        self.assertEqual(report["expected_total_cases"], 550)
        self.assertEqual(report["external_calls"], 0)
        self.assertEqual(report["bootstrap_unit"], "family_id")

    def test_plan_is_deterministic_and_selects_unique_sources(self):
        spec = load_spec(DEFAULT_SPEC)
        sources = load_source_cases()
        first = build_generation_plan(spec, sources)
        second = build_generation_plan(spec, sources)

        self.assertEqual(first, second)
        selected = [item["source_case_id"] for item in first["families"]]
        self.assertEqual(len(selected), len(set(selected)))

    def test_spec_rejects_inconsistent_total(self):
        spec = small_spec()
        spec["dataset"]["expected_total_cases"] = 14
        with self.assertRaisesRegex(ValueError, "expected_total_cases"):
            validate_spec(spec)


class RobustnessGenerationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            SourceCase("case_001", "Question one?", "Reference one."),
            SourceCase("case_002", "Question two?", "Reference two."),
        ]

    def test_preserving_variants_reuse_the_source_reference(self):
        cases = normalize_family_payload(
            self.sources[0], family_payload(self.sources[0]), "fake-generator"
        )
        preserving = {
            case["test_type"]: case for case in cases
            if case["expected_behavior"] == "preserve"
        }
        self.assertEqual(set(preserving), {"paraphrase", "lay_language", "distractor_noise"})
        self.assertTrue(
            all(case["reference_answer"] == "Reference one." for case in preserving.values())
        )
        self.assertTrue(all(case["paired_with"].endswith("_original") for case in preserving.values()))

    def test_duplicate_questions_are_rejected(self):
        payload = family_payload(self.sources[0])
        payload["variants"][0]["question"] = self.sources[0].question
        with self.assertRaisesRegex(ValueError, "duplicate questions"):
            normalize_family_payload(self.sources[0], payload, "fake-generator")

    def test_generation_checkpoints_and_resume_avoids_repeat_calls(self):
        spec = small_spec()
        plan = build_generation_plan(spec, self.sources)
        family_generator = FakeFamilyGenerator()
        conflict_generator = FakeConflictGenerator()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "robustness.json"
            dataset = generate_dataset(
                spec=spec,
                plan=plan,
                source_cases=self.sources,
                family_generator=family_generator,
                conflict_generator=conflict_generator,
                output=output,
                resume=False,
                delay_s=0,
            )
            self.assertEqual(dataset["summary"]["n_cases"], 15)
            self.assertEqual(dataset["summary"]["bootstrap_unit"], "family_id")
            self.assertEqual(len(family_generator.calls), 2)
            self.assertEqual(len(conflict_generator.calls), 1)
            self.assertEqual(len(dataset["metadata"]["dataset_fingerprint"]), 64)

            second_family_generator = FakeFamilyGenerator()
            second_conflict_generator = FakeConflictGenerator()
            resumed = generate_dataset(
                spec=spec,
                plan=plan,
                source_cases=self.sources,
                family_generator=second_family_generator,
                conflict_generator=second_conflict_generator,
                output=output,
                resume=True,
                delay_s=0,
            )
            self.assertEqual(second_family_generator.calls, [])
            self.assertEqual(second_conflict_generator.calls, [])
            self.assertEqual(
                resumed["metadata"]["dataset_fingerprint"],
                dataset["metadata"]["dataset_fingerprint"],
            )

            with output.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertTrue(saved["metadata"]["complete"])


if __name__ == "__main__":
    unittest.main()
