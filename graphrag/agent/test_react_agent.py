import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from graphrag.agent.react_agent import (
    MAX_RETRIES,
    _mandatory_retrieval_call,
    _mandatory_safety_call,
    validate_node,
)


class _Response:
    content = "RETRY: add missing evidence"


class _FakeLLM:
    def __init__(self, **kwargs):
        pass

    def invoke(self, messages):
        return _Response()


class ReactAgentStateTests(unittest.TestCase):
    def base_state(self):
        return {
            "query": "test question",
            "messages": [],
            "retrieved_context": [],
            "retry_count": 0,
            "final_answer": "",
            "candidate_answers": [],
            "validation_verdicts": [],
            "status": "running",
        }

    def test_mandatory_retrieval_starts_with_graph(self):
        call = _mandatory_retrieval_call(self.base_state())
        self.assertEqual(call.tool_calls[0]["name"], "retrieve_graph")

    def test_mandatory_retrieval_then_requires_vector(self):
        from langchain_core.messages import ToolMessage

        state = self.base_state()
        state["messages"] = [
            ToolMessage(content="graph context", tool_call_id="1", name="retrieve_graph")
        ]
        call = _mandatory_retrieval_call(state)
        self.assertEqual(call.tool_calls[0]["name"], "retrieve_vector")

    @patch("graphrag.agent.react_agent.find_known_drugs", return_value=["terlipressin"])
    def test_known_drug_forces_safety_check(self, find_drugs):
        from langchain_core.messages import ToolMessage

        state = self.base_state()
        state["messages"] = [
            ToolMessage(content="graph", tool_call_id="1", name="retrieve_graph"),
            ToolMessage(content="vector", tool_call_id="2", name="retrieve_vector"),
        ]
        call = _mandatory_safety_call(
            state, AIMessage(content="Terlipressin may be considered.")
        )
        self.assertEqual(call.tool_calls[0]["name"], "check_contraindications")
        self.assertEqual(call.tool_calls[0]["args"]["drug"], "terlipressin")

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI", _FakeLLM)
    def test_max_retry_preserves_candidate_answer(self):
        state = {
            "query": "test question",
            "messages": [AIMessage(content="grounded candidate")],
            "retrieved_context": ["supporting context"],
            "retry_count": MAX_RETRIES - 1,
            "final_answer": "",
            "candidate_answers": [],
            "validation_verdicts": [],
            "status": "running",
        }

        update = validate_node(state)

        self.assertEqual(update["final_answer"], "grounded candidate")
        self.assertEqual(update["status"], "max_retries_unapproved")
        self.assertEqual(update["retry_count"], MAX_RETRIES)
        self.assertEqual(update["candidate_answers"], ["grounded candidate"])

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_validator_prioritizes_recent_safety_context(self, llm_class):
        llm_class.return_value.invoke.return_value = _Response()
        state = self.base_state()
        state["messages"] = [AIMessage(content="candidate")]
        state["retrieved_context"] = ["v" * 10000, "SAFETY WARNING"]

        validate_node(state)

        prompt = llm_class.return_value.invoke.call_args.args[0][0].content
        self.assertIn("SAFETY WARNING", prompt)
        self.assertLess(prompt.index("SAFETY WARNING"), prompt.index("v" * 100))


if __name__ == "__main__":
    unittest.main()
