import json
import tempfile
import unittest
from pathlib import Path

from graphrag.eval.mirage_benchmark import (
    DATASET_ORDER,
    MirageCase,
    _run_textbooks_rag_case,
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
    def test_textbooks_rag_keeps_options_out_of_retrieval(self):
        from graphrag.eval.mirage_corpus import TextbookSnippet

        queries = []
        prompts = []

        class Retriever:
            def retrieve(self, query, k):
                queries.append((query, k))
                return [TextbookSnippet("doc-1", "Title", "Evidence", 1.0)]

        class LLM:
            def invoke(self, messages):
                prompts.append(messages[0].content)
                return type(
                    "Response",
                    (),
                    {"content": '{"answer_choice":"A"}', "usage_metadata": {}},
                )()

        class Message:
            def __init__(self, content):
                self.content = content

        case = MirageCase(
            case_id="mmlu:test",
            dataset="mmlu",
            source_id="test",
            retrieval_query="Pure medical question?",
            answer_query="Pure medical question?\nA. SECRET OPTION\nB. Other",
            options={"A": "SECRET OPTION", "B": "Other"},
            answer="A",
        )
        result = _run_textbooks_rag_case(case, Retriever(), LLM(), Message, 8)

        self.assertEqual(queries, [("Pure medical question?", 8)])
        self.assertNotIn("SECRET OPTION", queries[0][0])
        self.assertIn("SECRET OPTION", prompts[0])
        self.assertEqual(result["retrieved_ids"], ["doc-1"])

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
        self.assertEqual(
            parse_answer_choice(
                r"Option C is discussed, but the final answer is $\boxed{A}$", choices
            ),
            "A",
        )
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

    def test_imports_compatible_results_without_repeating_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.json"
            source.write_text(json.dumps(_fixture()), encoding="utf-8")
            cases = stratified_sample(
                load_benchmark(source, enforce_official_counts=False), 5, 2
            )
            baseline_output = Path(directory) / "baseline.json"
            target_output = Path(directory) / "comparison.json"

            def baseline(case):
                return {"answer": '{"answer_choice":"' + case.answer + '"}'}

            run_benchmark(
                cases,
                {"closed-book": baseline},
                baseline_output,
                source,
                generation_model="fake-model",
                resume=False,
            )
            candidate_calls = []

            def must_not_repeat_baseline(case):
                raise AssertionError("import repeated a paid baseline call")

            def candidate(case):
                candidate_calls.append(case.case_id)
                return {"answer": '{"answer_choice":"' + case.answer + '"}'}

            report = run_benchmark(
                cases,
                {
                    "closed-book": must_not_repeat_baseline,
                    "textbooks-rag": candidate,
                },
                target_output,
                source,
                generation_model="fake-model",
                resume=False,
                reuse_results_from=baseline_output,
            )

            self.assertEqual(len(candidate_calls), 5)
            self.assertEqual(report["reused_checkpoint"]["saved_model_calls"], 5)
            self.assertEqual(
                report["paired_comparisons"]["textbooks-rag_vs_closed-book"][
                    "n_pairs"
                ],
                5,
            )

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
                context_count=1 if system == "textbooks-rag" else 0,
            )

        baseline = [result("1", "closed-book", False), result("2", "closed-book", True)]
        agent = [result("1", "current-agent", True), result("2", "current-agent", True)]
        self.assertEqual(summarize_system(agent)["accuracy"], 1.0)
        priced = summarize_system(
            agent,
            {"input": 0.30, "output_including_thinking": 2.50},
        )
        self.assertEqual(priced["total_input_tokens"], 2)
        self.assertEqual(priced["total_output_tokens"], 2)
        self.assertAlmostEqual(priced["estimated_cost_usd"], 0.0000056)
        comparison = paired_comparison(baseline, agent)
        self.assertEqual(comparison["candidate_wins"], 1)
        self.assertEqual(comparison["baseline_wins"], 0)
        self.assertEqual(comparison["accuracy_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
