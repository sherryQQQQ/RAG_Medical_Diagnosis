import unittest

from graphrag.agent.smoke import summarize


class SmokeSummaryTests(unittest.TestCase):
    def test_summarize_separates_pipeline_and_keyword_checks(self):
        summary = summarize(
            [
                {
                    "pipeline_pass": True,
                    "keyword_pass": False,
                    "status": "approved",
                    "tool_errors": [],
                    "retry_count": 1,
                    "latency_s": 2.0,
                },
                {
                    "pipeline_pass": True,
                    "keyword_pass": True,
                    "status": "max_retries_unapproved",
                    "tool_errors": ["error"],
                    "retry_count": 3,
                    "latency_s": 4.0,
                },
            ]
        )
        self.assertEqual(summary["pipeline_passed"], 2)
        self.assertEqual(summary["keyword_passed"], 1)
        self.assertEqual(summary["approved"], 1)
        self.assertEqual(summary["tool_errors"], 1)
        self.assertEqual(summary["average_retries"], 2.0)
        self.assertEqual(summary["average_latency_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
