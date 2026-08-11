import tempfile
import unittest
from pathlib import Path

from graphrag.eval.robustness_benchmark import (
    DEFAULT_PREVIEW,
    RobustnessCase,
    RobustnessResult,
    SYSTEMS,
    _apply_output_guards,
    _summarize_judgments,
    _validated_judgment,
    build_pilot_dataset,
    load_pilot_cases,
    run_generation,
    save_pilot_dataset,
)


def _judgment_payload(case: RobustnessCase, pass_value: bool = True):
    preserving = case.test_type in {"paraphrase", "distractor_noise"}
    directional = case.test_type == "directional_expectation"
    conflict = case.test_type == "conflicting_evidence"
    failure = case.test_type == "vector_tool_failure"
    value = {
        "answer_correct": pass_value,
        "behavior_pass": pass_value,
        "abstained": case.test_type == "abstention" and pass_value,
        "consistent_with_original": "pass" if preserving else "not_applicable",
        "directionally_correct": "pass" if directional else "not_applicable",
        "conflict_handling": "pass" if conflict else "not_applicable",
        "tool_failure_recovery": "pass" if failure else "not_applicable",
        "unsupported_claim_count": 0,
        "reason": "fixture",
    }
    return {"systems": {system: dict(value) for system in SYSTEMS}}


def _result(case: RobustnessCase, system: str, answer: str) -> RobustnessResult:
    return RobustnessResult(
        case_id=case.case_id,
        family_id=case.family_id,
        test_type=case.test_type,
        expected_behavior=case.expected_behavior,
        system=system,
        raw_answer=answer,
        latency_s=1.0,
        status="completed",
        error="",
        tool_names=["retrieve_medical_evidence"],
        tool_errors=[],
        retry_count=0,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        provider_request_count=1,
    )


class RobustnessBenchmarkTests(unittest.TestCase):
    def test_pilot_reuses_preview_and_builds_six_slices_offline(self):
        dataset = build_pilot_dataset(DEFAULT_PREVIEW)
        self.assertEqual(dataset["metadata"]["case_count"], 30)
        self.assertEqual(dataset["metadata"]["external_generation_calls"], 0)
        self.assertEqual(dataset["metadata"]["counts"]["conflicting_evidence"], 5)
        self.assertEqual(dataset["metadata"]["counts"]["vector_tool_failure"], 5)
        self.assertEqual(dataset["metadata"]["counts"]["directional_expectation"], 1)
        self.assertEqual(dataset["metadata"]["counts"]["abstention"], 4)
        conflict = next(
            item for item in dataset["cases"] if item["test_type"] == "conflicting_evidence"
        )
        self.assertTrue(conflict["injected_context"][0].startswith("UNTRUSTED"))
        self.assertIn(conflict["forbidden_claims"][0], conflict["injected_context"][0])

    def test_judgment_validation_enforces_slice_applicability(self):
        dataset = build_pilot_dataset(DEFAULT_PREVIEW)
        case = RobustnessCase(
            **next(item for item in dataset["cases"] if item["test_type"] == "paraphrase")
        )
        payload = _judgment_payload(case)
        validated = _validated_judgment(payload, case)
        self.assertEqual(
            validated["systems"][SYSTEMS[1]]["consistent_with_original"],
            "pass",
        )
        payload["systems"][SYSTEMS[1]]["consistent_with_original"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "must be judged"):
            _validated_judgment(payload, case)

    def test_contract_normalization_fixes_cross_field_judge_contradictions(self):
        dataset = build_pilot_dataset(DEFAULT_PREVIEW)
        case = RobustnessCase(
            **next(
                item
                for item in dataset["cases"]
                if item["test_type"] == "distractor_noise"
            )
        )
        payload = _judgment_payload(case)
        for item in payload["systems"].values():
            item["abstained"] = True
            item["answer_correct"] = True
            item["behavior_pass"] = True
        normalized = _validated_judgment(payload, case)
        for item in normalized["systems"].values():
            self.assertTrue(item["raw_answer_correct"])
            self.assertFalse(item["answer_correct"])
            self.assertFalse(item["behavior_pass"])
            self.assertTrue(item["contract_corrected"])

    def test_saved_output_guard_requires_explicit_conflict_disclosure(self):
        dataset = build_pilot_dataset(DEFAULT_PREVIEW)
        case = RobustnessCase(
            **next(
                item
                for item in dataset["cases"]
                if item["test_type"] == "conflicting_evidence"
            )
        )
        payload = _validated_judgment(_judgment_payload(case), case)
        outputs = {
            system: _result(case, system, "The reference answer is correct.")
            for system in SYSTEMS
        }
        guarded = _apply_output_guards(payload, case, outputs)
        for item in guarded["systems"].values():
            self.assertEqual(item["raw_conflict_handling"], "pass")
            self.assertEqual(item["conflict_handling"], "fail")
            self.assertFalse(item["behavior_pass"])
            self.assertTrue(item["contract_corrected"])

    def test_generation_checkpoint_resume_does_not_repeat_completed_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.json"
            output_path = root / "results.json"
            save_pilot_dataset(dataset_path, DEFAULT_PREVIEW)
            calls = []

            def fake_runner(case):
                calls.append(case.case_id)
                return {
                    "answer": case.reference_answer,
                    "status": "completed",
                    "tool_names": ["retrieve_textbooks_bm25"],
                    "tool_errors": [],
                    "retrieved_ids": ["doc-1"],
                    "retrieval_latency_s": 0.01,
                    "context_count": 1,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "provider_request_count": 1,
                }

            runners = {system: fake_runner for system in SYSTEMS}
            report = run_generation(
                dataset_path,
                output_path,
                root / "unused.sqlite3",
                "fixture-model",
                runners=runners,
            )
            self.assertEqual(len(calls), 60)
            self.assertEqual(len(report["results"]), 60)
            run_generation(
                dataset_path,
                output_path,
                root / "unused.sqlite3",
                "fixture-model",
                runners=runners,
            )
            self.assertEqual(len(calls), 60)

    def test_summary_reports_requested_robustness_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            save_pilot_dataset(path, DEFAULT_PREVIEW)
            _, cases = load_pilot_cases(path)
        judgments = []
        for case in cases:
            judgments.append(
                {
                    "case_id": case.case_id,
                    "family_id": case.family_id,
                    "test_type": case.test_type,
                    **_judgment_payload(case),
                    "latency_s": 1.0,
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "error": "",
                }
            )
        generation = {
            "metrics": {
                system: {"estimated_cost_usd": 0.1}
                for system in SYSTEMS
            }
        }
        summary = _summarize_judgments(generation, judgments, None)
        agent = summary["systems"][SYSTEMS[1]]
        self.assertEqual(agent["paired_consistency"]["n"], 10)
        self.assertEqual(agent["paired_consistency"]["rate"], 1.0)
        self.assertEqual(agent["correctness_retention"]["rate"], 1.0)
        self.assertEqual(agent["directional_consistency"], {"n": 1, "rate": 1.0})
        self.assertEqual(agent["abstention"]["f1"], 1.0)
        self.assertEqual(agent["conflict_handling_rate"], 1.0)
        self.assertEqual(agent["tool_failure_recovery_rate"], 1.0)
        self.assertEqual(agent["worst_slice_behavior_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
