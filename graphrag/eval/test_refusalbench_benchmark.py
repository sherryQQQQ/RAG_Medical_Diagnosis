import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.refusalbench_benchmark import (
    ActionResult,
    JudgeResult,
    RefusalCase,
    call_and_cost_plan,
    load_selection,
    parse_action_response,
    run_generation,
    run_judgments,
)
from graphrag.eval.refusalbench_holdout import fingerprint


def selection_fixture(path: Path) -> dict:
    cases = [
        {
            "case_id": "answer-case",
            "pair_id": "pair-1",
            "source_id": "source-1",
            "query": "What is supported?",
            "grounding": ["The supported answer is alpha."],
            "reference_answer": "alpha",
            "expected_rag_behavior": "ANSWER_CORRECTLY",
            "expected_action": "answer",
            "answerable": True,
            "perturbation_class": "P-Ambiguity",
            "intensity": "LOW",
            "generator_model": "fixture",
            "verifier_votes": {"a": "PASS"},
            "review_status": "fixture",
        },
        {
            "case_id": "refuse-case",
            "pair_id": "pair-1",
            "source_id": "source-1",
            "query": "Which conflicting answer is correct?",
            "grounding": ["alpha", "not alpha"],
            "reference_answer": "alpha",
            "expected_rag_behavior": "REFUSE_CONTRADICTORY_CONTEXT",
            "expected_action": "escalate",
            "answerable": False,
            "perturbation_class": "P-Contradiction",
            "intensity": "HIGH",
            "generator_model": "fixture",
            "verifier_votes": {"a": "PASS"},
            "review_status": "fixture",
        },
    ]
    payload = {
        "schema_version": 1,
        "selection_fingerprint": fingerprint(cases),
        "scope": "fixture",
        "cases": cases,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class RefusalBenchBenchmarkTests(unittest.TestCase):
    def test_action_parser_requires_valid_action_and_citation(self):
        parsed = parse_action_response(
            '{"action":"answer","answer":"alpha","reason":"supported",'
            '"cited_evidence_ids":["source-1"]}'
        )
        self.assertEqual(parsed["action"], "answer")
        with self.assertRaisesRegex(ValueError, "cited"):
            parse_action_response(
                '{"action":"answer","answer":"alpha","reason":"supported",'
                '"cited_evidence_ids":[]}'
            )

    def test_generation_checkpoints_and_resumes_without_repeating_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            output = root / "results.json"
            selection_fixture(selection)
            calls = []

            def runner(case):
                calls.append(case.case_id)
                action = "answer" if case.answerable else "escalate"
                return {
                    "action": action,
                    "answer": "alpha" if case.answerable else "clinician review",
                    "reason": "fixture",
                    "cited_evidence_ids": ["source-1"] if case.answerable else [],
                    "raw_output": "fixture",
                    "provider_request_count": 1,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }

            runners = {name: runner for name in ("direct-rag", "agent-v2", "agent-v3")}
            first = run_generation(
                selection_path=selection,
                output_path=output,
                model="gemini-2.5-flash",
                runners=runners,
            )
            self.assertEqual(len(calls), 6)
            second = run_generation(
                selection_path=selection,
                output_path=output,
                model="gemini-2.5-flash",
                runners=runners,
            )
        self.assertEqual(len(calls), 6)
        self.assertEqual(first, second)
        self.assertEqual(first["metrics"]["agent-v3"]["exact_action_accuracy"], 1.0)
        self.assertEqual(first["metrics"]["agent-v3"]["safe_deferral"]["recall"], 1.0)
        self.assertEqual(first["metrics"]["agent-v3"]["per_action"]["escalate"]["recall"], 1.0)
        self.assertEqual(
            first["paired_exact_action"]["direct-rag_vs_agent-v3"]["ties"], 2
        )

    def test_judge_only_scores_answerable_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            generation = root / "generation.json"
            judgments = root / "judgments.json"
            payload = selection_fixture(selection)
            results = []
            for case in payload["cases"]:
                for system in ("direct-rag", "agent-v2", "agent-v3"):
                    action = "answer" if case["answerable"] else "escalate"
                    if case["answerable"] and system == "agent-v3":
                        action = "abstain"
                    results.append(
                        ActionResult(
                            case_id=case["case_id"],
                            pair_id=case["pair_id"],
                            source_id=case["source_id"],
                            system=system,
                            expected_rag_behavior=case["expected_rag_behavior"],
                            expected_action=case["expected_action"],
                            expected_answerable=case["answerable"],
                            action=action,
                            answer="alpha",
                            reason="fixture",
                            cited_evidence_ids=["source-1"],
                            raw_output="alpha",
                            status="completed",
                            error="",
                            latency_s=0.1,
                            input_tokens=1,
                            output_tokens=1,
                            total_tokens=2,
                            provider_request_count=1,
                        )
                    )
            generation.write_text(
                json.dumps(
                    {
                        "selection_fingerprint": payload["selection_fingerprint"],
                        "results": [result.__dict__ for result in results],
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def judge(case, outputs):
                calls.append(case.case_id)
                return JudgeResult(
                    case_id=case.case_id,
                    systems={
                        system: {"correct": True, "reason": "fixture"}
                        for system in outputs
                    },
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    latency_s=0.1,
                )

            first = run_judgments(
                selection_path=selection,
                generation_path=generation,
                output_path=judgments,
                model="gemini-2.5-flash",
                runner=judge,
            )
            second = run_judgments(
                selection_path=selection,
                generation_path=generation,
                output_path=judgments,
                model="gemini-2.5-flash",
                runner=judge,
            )
        self.assertEqual(calls, ["answer-case"])
        self.assertEqual(first, second)
        self.assertEqual(first["answer_accuracy"]["direct-rag"]["answerable_accuracy"], 1.0)
        self.assertEqual(first["answer_accuracy"]["agent-v3"]["answerable_accuracy"], 0.0)
        self.assertEqual(len(first["deterministic_contract_overrides"]), 1)

    def test_judge_checkpoint_retries_only_evaluator_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            generation = root / "generation.json"
            judgments = root / "judgments.json"
            payload = selection_fixture(selection)
            results = []
            for case in payload["cases"]:
                for system in ("direct-rag", "agent-v2", "agent-v3"):
                    results.append(
                        ActionResult(
                            case_id=case["case_id"],
                            pair_id=case["pair_id"],
                            source_id=case["source_id"],
                            system=system,
                            expected_rag_behavior=case["expected_rag_behavior"],
                            expected_action=case["expected_action"],
                            expected_answerable=case["answerable"],
                            action="answer" if case["answerable"] else "escalate",
                            answer="alpha",
                            reason="fixture",
                            cited_evidence_ids=["source-1"],
                            raw_output="alpha",
                            status="completed",
                            error="",
                            latency_s=0.1,
                            input_tokens=1,
                            output_tokens=1,
                            total_tokens=2,
                            provider_request_count=1,
                        )
                    )
            generation.write_text(
                json.dumps(
                    {
                        "selection_fingerprint": payload["selection_fingerprint"],
                        "results": [result.__dict__ for result in results],
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def judge(case, outputs):
                calls.append(case.case_id)
                if len(calls) == 1:
                    return JudgeResult(
                        case_id=case.case_id,
                        systems={},
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        latency_s=0.1,
                        error="invalid_judge_output:truncated",
                    )
                return JudgeResult(
                    case_id=case.case_id,
                    systems={
                        system: {"correct": True, "reason": "fixture"}
                        for system in outputs
                    },
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    latency_s=0.1,
                )

            first = run_judgments(
                selection_path=selection,
                generation_path=generation,
                output_path=judgments,
                model="gemini-2.5-flash",
                runner=judge,
            )
            second = run_judgments(
                selection_path=selection,
                generation_path=generation,
                output_path=judgments,
                model="gemini-2.5-flash",
                runner=judge,
            )
        self.assertEqual(calls, ["answer-case", "answer-case"])
        self.assertEqual(first["answer_accuracy"]["agent-v3"]["n"], 0)
        self.assertEqual(second["answer_accuracy"]["agent-v3"]["n"], 1)

    def test_dry_run_plan_is_zero_call_and_hard_capped(self):
        cases = [
            RefusalCase(
                case_id=f"case-{index}",
                pair_id="pair",
                source_id="source",
                query="Question?",
                grounding=("Evidence",),
                reference_answer="Answer",
                expected_rag_behavior=(
                    "ANSWER_CORRECTLY" if index == 0 else "REFUSE_AMBIGUOUS_QUERY"
                ),
                expected_action="answer" if index == 0 else "clarify",
                answerable=index == 0,
                perturbation_class="fixture",
                intensity="LOW",
            )
            for index in range(2)
        ]
        plan = call_and_cost_plan(cases, "gemini-2.5-flash")
        self.assertEqual(plan["external_model_calls"], 0)
        self.assertEqual(plan["request_plan"]["total"]["expected"], 11)
        self.assertEqual(plan["request_plan"]["total"]["hard_cap"], 21)
        self.assertLessEqual(
            plan["token_cost_estimate"]["expected_cost_usd"],
            plan["token_cost_estimate"]["hard_cap_cost_usd"],
        )


if __name__ == "__main__":
    unittest.main()
