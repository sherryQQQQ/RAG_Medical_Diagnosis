import unittest

from graphrag.kg.builder import InvalidExtraction, normalize_extraction


class BuilderValidationTests(unittest.TestCase):
    def test_normalizes_and_deduplicates(self):
        extraction = normalize_extraction(
            {
                "entities": [
                    {"name": "  Heart Failure ", "type": "Disease"},
                    {"name": "heart failure", "type": "Disease"},
                ],
                "relations": [
                    {
                        "from": "Heart Failure",
                        "from_type": "Disease",
                        "relation": "TREATED_BY",
                        "to": "Medical therapy",
                        "to_type": "Treatment",
                    }
                ],
            }
        )
        self.assertEqual(len(extraction["entities"]), 2)
        self.assertEqual(extraction["relations"][0]["from"], "heart failure")

    def test_reverses_valid_backward_relation(self):
        extraction = normalize_extraction(
            {
                "entities": [],
                "relations": [
                    {
                        "from": "aspirin",
                        "from_type": "Drug",
                        "relation": "REQUIRES_DRUG",
                        "to": "antiplatelet therapy",
                        "to_type": "Treatment",
                    }
                ],
            }
        )
        relation = extraction["relations"][0]
        self.assertEqual(relation["from_type"], "Treatment")
        self.assertEqual(relation["to_type"], "Drug")

    def test_rejects_untrusted_schema_value(self):
        with self.assertRaises(InvalidExtraction):
            normalize_extraction(
                {"entities": [{"name": "x", "type": "Disease) MATCH (n) DETACH DELETE n"}]}
            )

    def test_coerces_endpoint_labels_for_whitelisted_relation(self):
        extraction = normalize_extraction(
            {
                "entities": [],
                "relations": [
                    {
                        "from": "hypertension",
                        "from_type": "Symptom",
                        "relation": "TREATED_BY",
                        "to": "medical therapy",
                        "to_type": "Treatment",
                    }
                ],
            }
        )
        relation = extraction["relations"][0]
        self.assertEqual((relation["from_type"], relation["to_type"]), ("Disease", "Treatment"))
        self.assertTrue(extraction["corrections"])

    def test_accepts_empty_summary_chunk(self):
        extraction = normalize_extraction({"entities": [], "relations": []})
        self.assertEqual(extraction["entities"], [])
        self.assertEqual(extraction["relations"], [])


if __name__ == "__main__":
    unittest.main()
