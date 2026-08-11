import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.mirage_benchmark import (
    DATASET_ORDER,
    load_benchmark,
    paired_comparison,
    parse_answer_choice,
    run_benchmark,
    stratified_sample,
    summarize_system,
)


def _fixture() -> dict:
    return {
        dataset: {
            f"id-{index}": {
                "question": f"Question {dataset} {index}?",
                "options": {"A": "First", "B": "Second"},
                "answer": "A" if index % 2 == 0 else "B",
            }
            for index in range(3)
        }
        for dataset in DATASET_ORDER
    }


class MirageBenchmarkTests(unittest.TestCase):
    def test_load_and_stratify_separates_retrieval_from_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(json.dumps(_fixture()), encoding="utf-8")
            cases = load_benchmark(path, enforce_official_counts=False)
            selected = stratified_sample(cases, limit=10, seed=13)

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            {name: sum(case.dataset == name for case in selected) for name in DATASET_ORDER},
            {name: 2 for name in DATASET_ORDER},
        )
        self.assertNotIn("First", selected[0].retrieval_query)
        self.assertIn("A. First", selected[0].answer_query)

    def test_answer_parser_requires_an_explicit_choice(self):
        choices = {"A", "B", "C", "D"}
        self.assertEqual(parse_answer_choice('{"answer_choice":"B"}', choices), "B")
        self.assertEqual(parse_answer_choice("Reasoning. FINAL_ANSWER: C", choices), "C")
        self.assertEqual(parse_answer_choice("**(A)**", choices), "A")
        self.assertEqual(parse_answer_choice(r"The final answer is $\boxed{D}$", choices), "D")
        self.assertEqual(parse_answer_choice("Aspirin may help.", choices), "")

    def test_run_checkpoints_and_resume_reuses_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.json"
            source.write_text(json.dumps(_fixture()), encoding="utf-8")
            cases = stratified_sample(
                load_benchmark(source, enforce_official_counts=False), 5, 2
            )
            output = Path(directory) / "results.json"
            calls = []

            def runner(case):
                calls.append(case.case_id)
                return {"answer": '{"answer_choice":"' + case.answer + '"}'}

            report = run_benchmark(
                cases,
                {"closed-book": runner},
                output,
                source,
                generation_model="fake-model",
                resume=False,
            )
            self.assertEqual(report["metrics"]["closed-book"]["accuracy"], 1.0)
            self.assertEqual(len(calls), 5)

            saved = json.loads(output.read_text(encoding="utf-8"))
            saved["answer_parser_version"] = 1
            saved["results"][0]["prediction"] = ""
            saved["results"][0]["correct"] = False
            output.write_text(json.dumps(saved), encoding="utf-8")

            def must_not_run(case):
                raise AssertionError("resume repeated a paid call")

            resumed = run_benchmark(
                cases,
                {"closed-book": must_not_run},
                output,
                source,
                generation_model="fake-model",
                resume=True,
            )
            self.assertEqual(resumed["metrics"]["closed-book"]["accuracy"], 1.0)

    def test_summary_and_paired_mcnemar_counts(self):
        from graphrag.eval.mirage_benchmark import SystemResult

        def result(case_id, system, correct):
            return SystemResult(
                case_id=case_id,
                dataset="mmlu",
                source_id=case_id,
                system=system,
                gold_choice="A",
                prediction="A" if correct else "B",
                correct=correct,
                raw_answer="",
                latency_s=1.0,
                status="completed",
                error="",
                tool_names=[],
                tool_errors=[],
                retry_count=0,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )

        baseline = [result("1", "closed-book", False), result("2", "closed-book", True)]
        agent = [result("1", "current-agent", True), result("2", "current-agent", True)]
        self.assertEqual(summarize_system(agent)["accuracy"], 1.0)
        comparison = paired_comparison(baseline, agent)
        self.assertEqual(comparison["candidate_wins"], 1)
        self.assertEqual(comparison["baseline_wins"], 0)
        self.assertEqual(comparison["accuracy_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
