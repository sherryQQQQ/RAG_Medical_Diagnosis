import unittest
from tempfile import TemporaryDirectory

from graphrag.eval.synthetic_compare import (
    HybridGraphVectorRetriever,
    build_synthetic_cases,
    compare_methods,
    generate_synthetic_queries,
    load_original_dataset,
    load_synthetic_dataset,
    save_synthetic_dataset,
)


class SyntheticCompareTests(unittest.TestCase):
    def test_generate_synthetic_queries(self):
        cases = build_synthetic_cases()
        queries = generate_synthetic_queries(cases, n_per_case=2, seed=1)

        self.assertEqual(len(queries), len(cases) * 2)
        self.assertEqual(len(cases), len(load_original_dataset()))
        self.assertTrue(all(query.query for query in queries))
        self.assertTrue(all(query.expected_disease for query in queries))

    def test_save_and_load_synthetic_dataset(self):
        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/synthetic.json"
            save_synthetic_dataset(path, n_per_case=2, seed=1)
            cases, queries = load_synthetic_dataset(path)

        self.assertEqual(len(cases), len(build_synthetic_cases()))
        self.assertEqual(len(queries), len(cases) * 2)
        self.assertIn("65-year-old man", cases[0].source_prompt)

    def test_retrievers_return_ranked_results(self):
        cases = build_synthetic_cases()
        queries = generate_synthetic_queries(cases[:5], n_per_case=2, seed=1)
        query = queries[0]

        hybrid_results = HybridGraphVectorRetriever(cases).retrieve(query.query, k=3)

        self.assertEqual(len(hybrid_results), 3)
        self.assertTrue(any(result.disease == query.expected_disease for result in hybrid_results))

    def test_hybrid_beats_vector_on_synthetic_set(self):
        with TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/synthetic.json"
            save_synthetic_dataset(path, n_per_case=6, seed=7)
            report = compare_methods(data_path=path)

        self.assertGreaterEqual(report.hybrid_graph_vector.mrr, report.vector_only.mrr)
        self.assertGreaterEqual(report.hybrid_graph_vector.top1, report.vector_only.top1)
        self.assertGreaterEqual(report.hybrid_graph_vector.top3, report.vector_only.top3)
        self.assertGreaterEqual(report.hybrid_graph_vector.top5, report.vector_only.top5)


if __name__ == "__main__":
    unittest.main()
