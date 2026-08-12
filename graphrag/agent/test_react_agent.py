import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from graphrag.agent.react_agent import (
    MAX_RETRIES,
    POLICY_VALIDATE_PROMPT,
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

    def test_benchmark_retrieval_query_excludes_answer_options(self):
        from langchain_core.messages import ToolMessage

        state = self.base_state()
        state["query"] = "Question?\nA. option one\nB. option two"
        state["retrieval_query"] = "Question?"
        graph_call = _mandatory_retrieval_call(state)
        self.assertEqual(
            graph_call.tool_calls[0]["args"]["symptoms"], '["Question?"]'
        )

        state["messages"] = [
            ToolMessage(content="graph", tool_call_id="1", name="retrieve_graph")
        ]
        vector_call = _mandatory_retrieval_call(state)
        self.assertEqual(vector_call.tool_calls[0]["args"]["query"], "Question?")

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

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_multiple_choice_policy_preserves_valid_explicit_choice_without_call(
        self, llm_class
    ):
        state = self.base_state()
        state.update(
            {
                "messages": [AIMessage(content='{"answer_choice": "B"}')],
                "task_mode": "multiple_choice",
                "allowed_choices": ["A", "B", "C", "D"],
            }
        )

        update = validate_node(state, enable_policy_precheck=True)

        llm_class.assert_not_called()
        self.assertEqual(update["final_answer"], '{"answer_choice": "B"}')
        self.assertEqual(update["status"], "policy_approved")
        self.assertEqual(update["policy_approval_count"], 1)
        self.assertNotIn("provider_request_count", update)

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_multiple_choice_policy_accepts_evaluator_explicit_formats(self, llm_class):
        for candidate in ("**(A)**", "The final answer is C.", r"\boxed{D}"):
            with self.subTest(candidate=candidate):
                state = self.base_state()
                state.update(
                    {
                        "messages": [AIMessage(content=candidate)],
                        "task_mode": "multiple_choice",
                        "allowed_choices": ["A", "B", "C", "D"],
                    }
                )
                update = validate_node(state, enable_policy_precheck=True)
                self.assertEqual(update["status"], "policy_approved")
        llm_class.assert_not_called()

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_multiple_choice_policy_does_not_accept_choice_only_in_reasoning(
        self, llm_class
    ):
        state = self.base_state()
        state.update(
            {
                "messages": [AIMessage(content="A is tempting, but the evidence is mixed.")],
                "task_mode": "multiple_choice",
                "allowed_choices": ["A", "B", "C", "D"],
            }
        )

        update = validate_node(
            state,
            validation_prompt_template=POLICY_VALIDATE_PROMPT,
            enable_policy_precheck=True,
        )

        llm_class.assert_not_called()
        self.assertEqual(update["status"], "retrying")
        self.assertNotIn("provider_request_count", update)
        self.assertIn("abstention is not allowed", update["validation_verdicts"][0])

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_tool_failure_policy_accepts_safe_disclosure_without_call(self, llm_class):
        state = self.base_state()
        state.update(
            {
                "messages": [
                    AIMessage(
                        content=(
                            "The retrieval tool failed, so I cannot determine a safe "
                            "regimen; seek clinician review."
                        )
                    )
                ],
                "retrieved_context": ["Tool unavailable (RuntimeError)."],
                "task_mode": "open_medical",
            }
        )

        update = validate_node(state, enable_policy_precheck=True)

        llm_class.assert_not_called()
        self.assertEqual(update["status"], "policy_approved")
        self.assertIn("tool-failure disclosure", update["validation_verdicts"][0])

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_tool_failure_policy_accepts_passive_safe_abstention(self, llm_class):
        state = self.base_state()
        state.update(
            {
                "messages": [
                    AIMessage(
                        content=(
                            "Retrieval of medical evidence is unavailable. Therefore, "
                            "this question cannot be answered safely from the available "
                            "evidence."
                        )
                    )
                ],
                "retrieved_context": ["Tool unavailable (RuntimeError)."],
                "task_mode": "open_medical",
            }
        )

        update = validate_node(state, enable_policy_precheck=True)

        llm_class.assert_not_called()
        self.assertEqual(update["status"], "policy_approved")

    @patch("graphrag.agent.react_agent.ChatGoogleGenerativeAI")
    def test_policy_prompt_includes_open_question_contract(self, llm_class):
        llm_class.return_value.invoke.return_value = _Response()
        state = self.base_state()
        state.update(
            {
                "messages": [AIMessage(content="generic guideline answer")],
                "task_mode": "open_medical",
                "task_policy": "Abstain when decisive patient data is missing.",
            }
        )

        validate_node(
            state,
            validation_prompt_template=POLICY_VALIDATE_PROMPT,
            enable_policy_precheck=True,
        )

        prompt = llm_class.return_value.invoke.call_args.args[0][0].content
        self.assertIn("Abstain when decisive patient data is missing.", prompt)


if __name__ == "__main__":
    unittest.main()
