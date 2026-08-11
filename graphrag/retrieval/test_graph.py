import unittest

from graphrag.retrieval.graph import GraphRetriever


class _Session:
    def __init__(self, records):
        self.records = records
        self.last_query = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        self.last_query = query
        return self.records


class _Driver:
    def __init__(self, records):
        self.fake_session = _Session(records)

    def session(self):
        return self.fake_session


class GraphRetrieverTests(unittest.TestCase):
    def make_retriever(self, records):
        retriever = object.__new__(GraphRetriever)
        retriever._driver = _Driver(records)
        retriever._relationship_types = None
        return retriever

    def test_contraindication_uses_directed_schema_and_distinct_result(self):
        retriever = self.make_retriever(
            [{"disease": "ischemia", "drug": "terlipressin"}]
        )
        warnings = retriever.check_contraindications(
            "Terlipressin", [" Recent Cardiac Ischemia "]
        )

        self.assertEqual(warnings, ["terlipressin contraindicated for ischemia"])
        self.assertIn("(d:Disease)-[:CONTRAINDICATED_WITH]->(dr:Drug)", retriever._driver.fake_session.last_query)
        self.assertIn("RETURN DISTINCT", retriever._driver.fake_session.last_query)
        self.assertIn(
            "condition CONTAINS toLower(d.name)",
            retriever._driver.fake_session.last_query,
        )

    def test_empty_conditions_do_not_query_database(self):
        retriever = self.make_retriever([])
        self.assertEqual(