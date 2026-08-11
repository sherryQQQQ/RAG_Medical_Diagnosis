import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from graphrag.eval.mirage_benchmark import (
    DATASET_ORDER,
    SystemResult,
    _sha256,
    dataset_fingerprint,
    load_benchmark,
    stratified_sample,
)
from graphrag.eval.mirage_corpus import (
    TextbooksBM25Retriever,
    build_bm25_index,
    load_manifest,
)
from graphrag.eval.mirage_judge import (
    JudgeResponse,
    _normalize_failure_categories,
    _validated_payload,
    run_judging,
)


class MirageJudgeTests(unittest.TestCase):
    def _fixture(self, directory: str):
        root = Path(directory)
        dataset = root / "benchmark.json"
        dataset.write_text(
            json.dumps(
                {
                    name: {
                        f"id-{index}": {
                            "question": f"Which therapy treats condition {name} {index}?",
                            "options": {"A": "Therapy alpha", "B": "Therapy beta"},
                            "answer": "A",
                        }
                        for index in range(3)
                    }
                    for name in DATASET_ORDER
                }
            ),
            encoding="utf-8",
        )

        corpus = root / "corpus"
        source = corpus / "chunk" / "fixture.jsonl"
        source.parent.mkdir(parents=True)
        rows = [
            {
                "id": f"fixture_{index}",
                "title": "Medicine",
                "content": (
                    f"Condition {name} is treated by therapy alpha, not therapy beta."
                ),
            }
            for index, name in enumerate(DATASET_ORDER)
        ]
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "name": "fixture/textbooks",
                    "revision": "test-revision",
                    "source": "https://example.test",
                    "files": [
                        {
                            "path": "chunk/fixture.jsonl",
                            "size": source.stat().st_size,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        index = root / "index.sqlite3"
        build_bm25_index(corpus, index, load_manifest(manifest_path))

        cases = stratified_sample(
            load_benchmark(dataset, enforce_official_counts=False), 5, 2
        )
        retriever = TextbooksBM25Retriever(index)
        results = []
        for case in cases:
            retrieved_ids = [
                item.snippet_id for item in retriever.retrieve(case.retrieval_query, 8)
            ]
            for system in ("textbooks-rag", "textbooks-agent"):
                results.append(
                    asdict(
                        SystemResult(
                            case_id=case.case_id,
                            dataset=case.dataset,
                            source_id=case.source_id,
                            system=system,
                            gold_choice="A",
                            prediction="A",
                            correct=True,
                            raw_answer='{"answer_choice":"A"}',
                            latency_s=1.0,
                            status="completed",
                            error="",
                            tool_names=["retrieve_textbooks_bm25"],
                            tool_errors=[],
                            retry_count=0,
                            input_tokens=10,
                            output_tokens=2,
                            total_tokens=12,
                            retrieved_ids=retrieved_ids,
                            context_count=len(retrieved_ids),
                        )
                    )
                )
        report_path = root / "generation.json"
        report_path.write_text(
            json.dumps(
                {
                    "benchmark": {
                        "source_sha256": _sha256(dataset),
                        "question_only_retrieval": True,
                        "n_selected": 5,
                        "selection_seed": 2,
                        "selection_fingerprint": dataset_fingerprint(cases),
                    },
                    "systems": ["closed-book", "textbooks-rag", "textbooks-agent"],
                    "results": results,
                }
            ),
            encoding="utf-8",
        )
        return dataset, index, report_path, root / "judgments.json"

    def test_payload_validation_rejects_impossible_relevant_count(self):
        payload = {
            "retrieval": {
                "evidence_sufficient": True,
                "required_facts_covered": 1,
                "required_facts_total": 1,
                "relevant_context_count": 9,
                "outdated_corpus": False,
            },
            "generations": {},
        }
        with self.assertRaisesRegex(ValueError, "relevant-context"):
            _validated_payload(payload, context_count=8)

    def test_incorrect_none_category_gets_deterministic_failure_slice(self):
        generation = {
            system: {
                "faithful": True,
                "unsupported_claim_count": 0,
                "evidence_conflict": "none",
                "failure_category": "none",
                "reason": "No category supplied.",
            }
            for system in ("textbooks-rag", "textbooks-agent")
        }
        payload = {
            "retrieval": {
                "evidence_sufficient": True,
                "required_facts_covered": 1,
                "required_facts_total": 1,
                "judged_recall_at_8": 1.0,
                "relevant_context_count": 1,
                "outdated_corpus": False,
                "reason": "Evidence is present.",
            },
            "generations": generation,
        }
        outputs = {
            system: SystemResult(
                case_id="mmlu:test",
                dataset="mmlu",
                source_id="test",
                system=system,
                gold_choice="A",
                prediction="B",
                correct=False,
                raw_answer='{"answer_choice":"B"}',
                latency_s=1.0,
                status="completed",
                error="",
                tool_names=["retrieve_textbooks_bm25"],
                tool_errors=[],
                retry_count=0,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
            )
            for system in ("textbooks-rag", "textbooks-agent")
        }
        _normalize_failure_categories(payload, outputs)
        self.assertEqual(
            payload["generations"]["textbooks-agent"]["failure_category"],
            "generation_failure",
        )

    def test_judge_checkpoints_and_resume_reuses_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, index, report_path, output = self._fixture(directory)
            calls = []

            def runner(case, context, results):
                calls.append(case.case_id)
                self.assertIn("therapy alpha", context.lower())
                self.assertEqual(set(results), {"textbooks-rag", "textbooks-agent"})
                generation = {
                    system: {
                        "faithful": True,
                        "unsupported_claim_count": 0,
                        "evidence_conflict": "none",
                        "failure_category": "none",
                        "reason": "Supported.",
                    }
                    for system in ("textbooks-rag", "textbooks-agent")
                }
                return JudgeResponse(
                    payload={
                        "retrieval": {
                            "evidence_sufficient": True,
                            "required_facts_covered": 1,
                            "required_facts_total": 1,
                            "relevant_context_count": 1,
                            "outdated_corpus": False,
                            "reason": "One supporting passage.",
                        },
                        "generations": generation,
                    },
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                )

            report = run_judging(
                report_path,
                dataset,
                index,
                output,
                "gemini-2.5-flash",
                runner=runner,
                resume=False,
                enforce_official_counts=False,
            )
            self.assertEqual(len(calls), 5)
            self.assertEqual(
                report["metrics"]["retrieval"]["evidence_sufficiency_rate"], 1.0
            )
            self.assertEqual(
                report["metrics"]["generation"]["textbooks-agent"][
                    "faithfulness_rate"
                ],
                1.0,
            )
            self.assertEqual(
                report["metrics"]["generation"]["textbooks-agent"][
                    "accuracy_when_evidence_sufficient"
                ],
                1.0,
            )
            self.assertGreater(
                report["metrics"]["judge_system"]["estimated_cost_usd"], 0
            )

            def must_not_run(case, context, results):
                raise AssertionError("resume repeated a paid judge call")

            resumed = run_judging(
                report_path,
                dataset,
                index,
                output,
                "gemini-2.5-flash",
                runner=must_not_run,
                resume=True,
                enforce_official_counts=False,
            )
            self.assertEqual(len(resumed["judgments"]), 5)

    def test_provider_error_is_sliced_as_generation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, index, report_path, output = self._fixture(directory)
            generation = json.loads(report_path.read_text(encoding="utf-8"))
            agent = next(
                item
                for item in generation["results"]
                if item["system"] == "textbooks-agent"
            )
            failed_case_id = agent["case_id"]
            agent["error"] = "ServerError: 504"
            agent["status"] = "error"
            agent["retrieved_ids"] = []
            report_path.write_text(json.dumps(generation), encoding="utf-8")

            def runner(case, context, results):
                return JudgeResponse(
                    payload={
                        "retrieval": {
                            "evidence_sufficient": True,
                            "required_facts_covered": 1,
                            "required_facts_total": 1,
                            "relevant_context_count": 1,
                            "outdated_corpus": False,
                            "reason": "Supported.",
                        },
                        "generations": {
                            system: {
                                "faithful": True,
                                "unsupported_claim_count": 0,
                                "evidence_conflict": "none",
                                "failure_category": "none",
                                "reason": "Supported.",
                            }
                            for system in ("textbooks-rag", "textbooks-agent")
                        },
                    }
                )

            judged = run_judging(
                report_path,
                dataset,
                index,
                output,
                "gemini-2.5-flash",
                runner=runner,
                resume=False,
                enforce_official_counts=False,
            )
            item = next(
                value
                for value in judged["judgments"]
                if value["case_id"] == failed_case_id
            )
            self.assertEqual(
                item["generations"]["textbooks-agent"]["failure_category"],
                "generation_failure",
            )


if __name__ == "__main__":
    unittest.main()
