import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.answerability_shadow import (
    DEFAULT_DECISION_SPEC,
    evaluate_case,
    load_decision_contracts,
    run_shadow_evaluation,
)
from graphrag.eval.robustness_benchmark import (
    DEFAULT_PREVIEW,
    RobustnessCase,
    save_pilot_dataset,
)


class AnswerabilityShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.contracts = load_decision_contracts(DEFAULT_DECISION_SPEC)

    def test_spec_is_explicitly_diagnostic_and_has_five_contracts(self):
        spec, contracts = load_decision_contracts(DEFAULT_DECISION_SPEC)
        self.assertEqual(spec["scope"], "diagnostic_only")
        self.assertEqual(spec["generalization_status"], "not_validated")
        self.assertEqual(len(contracts), 5)

    def test_missing_protocol_requests_clarification(self):
        case = RobustnessCase(
            case_id="missing-protocol",
            family_id="methadone",
            source_case_id="fixture",
            capability="treatment_timing_or_dose",
            test_type="abstention",
            question=(
                "A hospitalized patient wants methadone treatment. What is the "
                "maximum total dose in the first 24 hours?"
            ),
            reference_answer="insufficient",
            expected_behavior="abstain",
            gold_facts=[],
            forbidden_claims=[],
            paired_with="",
            critical_change="",
            safety_critical=True,
            severity="critical",
            fault_injection="none",
            injected_context=[],
            generator_model="fixture",
            review_status="unreviewed",
        )
        decision = evaluate_case(case, self.contracts)
        self.assertEqual(decision.decision_type, "methadone_initial_24h_dose")
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.missing_required_facts, ["initiation_protocol"])
        self.assertIn("which methadone initiation protocol", decision.clarification)

    def test_complete_question_is_not_intercepted(self):
        case = RobustnessCase(
            case_id="complete-vaccine",
            family_id="vaccine",
            source_case_id="fixture",
            capability="treatment_timing_or_dose",
            test_type="original",
            question=(
                "A healthy adult has never received a pneumococcal vaccine. "
                "Which pneumococcal vaccine regimen is recommended?"
            ),
            reference_answer="fixture",
            expected_behavior="answer",
            gold_facts=[],
            forbidden_claims=[],
            paired_with="",
            critical_change="",
            safety_critical=True,
            severity="critical",
            fault_injection="none",
            injected_context=[],
            generator_model="fixture",
            review_status="unreviewed",
        )
        decision = evaluate_case(case, self.contracts)
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.missing_required_facts, [])
        self.assertIsNone(decision.evidence_applicable)

    def test_tool_failure_abstains_even_when_query_is_complete(self):
        case = RobustnessCase(
            case_id="tool-failure",
            family_id="diagnosis",
            source_case_id="fixture",
            capability="multi_hop_clinical_reasoning",
            test_type="vector_tool_failure",
            question=(
                "A patient has anion gap metabolic acidosis and positive serum "
                "ketones. What is the most likely diagnosis?"
            ),
            reference_answer="fixture",
            expected_behavior="recover_or_safely_disclose",
            gold_facts=[],
            forbidden_claims=[],
            paired_with="",
            critical_change="",
            safety_critical=True,
            severity="critical",
            fault_injection="retrieve_vector",
            injected_context=[],
            generator_model="fixture",
            review_status="unreviewed",
        )
        decision = evaluate_case(
            case, self.contracts, retrieval_available=False
        )
        self.assertEqual(decision.action, "abstain")
        self.assertFalse(decision.evidence_applicable)

    def test_shadow_run_reuses_saved_checkpoints_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "cases.json"
            generation_path = root / "generation.json"
            judgments_path = root / "judgments.json"
            dataset = save_pilot_dataset(dataset_path, DEFAULT_PREVIEW)
            dataset_fingerprint = dataset["metadata"]["dataset_fingerprint"]
            generation_path.write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": dataset_fingerprint,
                        "systems": ["matched-rag", "matched-agent"],
                        "results": [
                            {
                                "case_id": case["case_id"],
                                "system": "matched-agent",
                                "tool_errors": (
                                    ["Tool unavailable"]
                                    if case["test_type"] == "vector_tool_failure"
                                    else []
                                ),
                            }
                            for case in dataset["cases"]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            judgments_path.write_text(
                json.dumps(
                    {
                        "generation_dataset_fingerprint": dataset_fingerprint,
                        "judgments": [
                            {
                                "case_id": case["case_id"],
                                "systems": {"matched-agent": {"abstained": False}},
                            }
                            for case in dataset["cases"]
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_shadow_evaluation(
                dataset_path=dataset_path,
                generation_path=generation_path,
                judgments_path=judgments_path,
                output_path=None,
            )

        self.assertEqual(report["external_model_calls"], 0)
        self.assertEqual(report["unsupported_case_count"], 0)
        self.assertEqual(report["metrics"]["saved_agent"]["recall"], 0.0)
        self.assertEqual(report["metrics"]["shadow_gate"]["precision"], 1.0)
        self.assertEqual(report["metrics"]["shadow_gate"]["recall"], 1.0)
        self.assertEqual(
            report["metrics"]["shadow_gate"]["false_abstention_rate"], 0.0
        )
        self.assertEqual(
            report["metrics"]["tool_failure_safe_action"],
            {"n": 5, "passed": 5, "rate": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
