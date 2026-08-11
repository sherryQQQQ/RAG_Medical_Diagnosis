import tempfile
import unittest
from pathlib import Path

from graphrag.eval.e2e_benchmark import (
    JUDGE_PROMPT,
    EvalCase,
    run_evaluation,
    summarize,
    token_f1,
)


def fake_runner(question):
    return {
        "answer": "Use the supported treatment.",
        "context": ["The guideline says to use the supported treatment."],
        "tool_names": ["retrieve_graph", "retrieve_vector"],
        "tool_errors": [],
        "retry_count": 1,
        "status": "approved",
        "validation_verdicts": ["APPROVED: grounded"],
        "trace_id": "trace-001",
    }


def fake_judge(question, reference, answer, context):
    return {
        "clinical_correctness": 5,
        "context_faithfulness": 5,
        "answer_relevance": 5,
        "completeness": 4,
        "medical_safety": 5,
        "unsupported_claims": 0,
        "unsafe": False,
        "reason": "Supported by the supplied context.",
    }


class EndToEndBenchmarkTests(unittest.TestCase):
    def test_token_f1(self):
        self.assertEqual(token_f1("aspirin", "aspirin"), 1.0)
        self.assertEqual(token_f1("aspirin", "warfarin"), 0.0)

    def test_judge_prompt_formats_json_example(self):
        prompt = JUDGE_PROMPT.format(
            question="Q",
            reference="R",
            context="C",
            answer="A",
        )
        self.assertIn('"clinical_correctness": 1', prompt)
        self.assertIn('"unsafe": false', prompt)

    def test_evaluation_writes_trace_and_summary(self):
        cases = [EvalCase("case_001", "What treatment?", "Use the supported treatment.")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            report = run_evaluation(
                cases, fake_runner, fake_judge, output, resume=False, judge_model="fake-judge"
            )
            self.assertTrue(output.exists())
            self.assertEqual(report["metrics"]["pipeline_success_rate"], 1.0)
            self.assertEqual(report["metrics"]["judge_success_rate"], 1.0)
            self.assertEqual(report["metrics"]["clinical_correctness"], 1.0)
            self.assertEqual(report["metrics"]["completeness"], 0.75)
            self.assertEqual(
                report["metrics"]["retrieval_tool_sequence_success_rate"], 1.0
            )
            self.assertEqual(report["metrics"]["reflection_approval_rate"], 1.0)
            self.assertEqual(report["cases"][0]["trace_id"], "trace-001")

    def test_judge_failure_retains_agent_trace(self):
        def failing_judge(question, reference, answer, context):
            raise ValueError("judge unavailable")

        cases = [EvalCase("case_001", "What treatment?", "Use treatment.")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            report = run_evaluation(
                cases,
                fake_runner,
                failing_judge,
                output,
                resume=False,
                judge_model="fake-judge",
            )
            self.assertEqual(report["metrics"]["pipeline_success_rate"], 1.0)
            self.assertEqual(report["metrics"]["judge_success_rate"], 0.0)
            self.assertEqual(report["cases"][0]["trace_id"], "trace-001")
            self.assertIn("judge unavailable", report["cases"][0]["judge_error"])

            def runner_must_not_repeat(question):
                raise AssertionError("resume should reuse the paid Agent answer")

            resumed = run_evaluation(
                cases,
                runner_must_not_repeat,
                fake_judge,
                output,
                resume=True,
                judge_model="fake-judge",
            )
            self.assertEqual(resumed["metrics"]["judge_success_rate"], 1.0)
            self.assertEqual(resumed["cases"][0]["trace_id"], "trace-001")
            self.assertEqual(resumed["cases"][0]["judge_error"], "")

    def test_empty_agent_answer_skips_judge(self):
        def empty_runner(question):
            return {
                "answer": "",
                "context": [],
                "tool_names": ["retrieve_graph", "retrieve_vector"],
                "trace_id": "trace-empty",
            }

        judge_calls = []

        def recording_judge(question, reference, answer, context):
            judge_calls.append(question)
            return fake_judge(question, reference, answer, context)

        cases = [EvalCase("case_001", "What treatment?", "Use treatment.")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            report = run_evaluation(
                cases,
                empty_runner,
                recording_judge,
                output,
                resume=False,
                judge_model="fake-judge",
            )
            self.assertEqual(judge_calls, [])
            self.assertEqual(report["metrics"]["pipeline_success_rate"], 0.0)
            self.assertIn("EmptyFinalAnswer", report["cases"][0]["agent_error"])

    def test_resume_rejects_different_dataset(self):
        first = [EvalCase("case_001", "Question one?", "Answer one.")]
        second = [EvalCase("case_002", "Question two?", "Answer two.")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            run_evaluation(
                first,
                fake_runner,
                fake_judge,
                output,
                resume=False,
                judge_model="fake-judge",
            )
            with self.assertRaisesRegex(ValueError, "different dataset"):
                run_evaluation(
                    second,
                    fake_runner,
                    fake_judge,
                    output,
                    resume=True,
                    judge_model="fake-judge",
                )

    def test_rejudge_allows_model_change_without_agent_rerun(self):
        cases = [EvalCase("case_001", "Question?", "Answer.")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            run_evaluation(
                cases,
                fake_runner,
                fake_judge,
                output,
                resume=False,
                judge_model="old-judge",
            )

            def runner_must_not_repeat(question):
                raise AssertionError("rejudge should reuse the Agent answer")

            with self.assertRaisesRegex(ValueError, "different judge model"):
                run_evaluation(
                    cases,
                    runner_must_not_repeat,
                    fake_judge,
                    output,
                    resume=True,
                    judge_model="new-judge",
                )

            report = run_evaluation(
                cases,
                runner_must_not_repeat,
                fake_judge,
                output,
                resume=True,
                judge_model="new-judge",
                allow_rejudge=True,
            )
            self.assertEqual(report["judge_model"], "new-judge")
            self.assertEqual(report["metrics"]["judge_success_rate"], 1.0)
            self.assertEqual(report["cases"][0]["trace_id"], "trace-001")

    def test_summarize_empty(self):
        self.assertEqual(summarize([])["pipeline_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
