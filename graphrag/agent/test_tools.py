import unittest

from graphrag.agent.tools import _parse_string_list


class ToolArgumentTests(unittest.TestCase):
    def test_parses_json_list(self):
        self.assertEqual(_parse_string_list('["fever", " cough "]'), ["fever", "cough"])

    def test_wraps_json_string(self):
        self.assertEqual(_parse_string_list('"hypertension"'), ["hypertension"])

    def test_falls_back_to_comma_separated_text(self):
        self.assertEqual(_parse_string_list("fever, cough"), ["fever", "cough"])

    def test_rejects_non_list_json_shape(self):
        self.assertEqual(_parse_string_list('{"symptom": "fever"}'), [])


if __name__ == "__main__":
    unittest.main()
