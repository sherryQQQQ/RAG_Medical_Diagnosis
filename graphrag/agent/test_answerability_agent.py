import unittest

from graphrag.agent.answerability_agent import (
    EvidenceApplicabilityDecision,
    InvalidGateDecision,
    QueryCompletenessDecision,
    RetrievalEvidence,
    build_answerability_agent,
    initial_answerability_state,
    parse_evidence_applicability,
    parse_query_completeness,
    route_answerability,
)


def complete_query(**overrides):
    payload = {
        "schema_version": 1,
        "decision_requested": "select a treatment",
        "known_patient_facts": ["diagnosis"],
        "required_patient_facts": ["diagnosis"],
        "missing_required_facts": [],
        "query_complete": True,
        "query_issue": "none",
        "risk_level": "high",
        "clarification_questions": [],
        "reason": "The diagnosis is stated.",
    }
    payload.update(overrides)
    return payload


def incomplete_query(**overrides):
    payload = {
        "schema_version": 1,
        "decision_requested": "select a treatment",
        "known_patient_facts": [],
        "required_patient_facts": ["diagnosis"],
        "missing_required_facts": ["diagnosis"],
        "query_complete": False,
        "query_issue": "missing_required_facts",
        "risk_level": "high",
        "clarification_questions": ["What diagnosis has been established?"],
        "reason": "The diagnosis is missing.",
    }
    payload.update(overrides)
    return payload


def applicable_evidence(**overrides):
    payload = {
        "schema_version": 1,
        "evidence_available": True,
        "sufficient": True,
        "patient_applicable": True,
        "conflicting": False,
        "outdated": False,
        "tool_failed": False,
        "cited_evidence_ids": ["doc-1"],
        "reason": "The passage supports this patient-specific decision.",
    }
    payload.update(overrides)
    return payload


class StructuredGateTests(unittest.TestCase):
    def test_query_gate_requires_known_and_missing_partition(self):
        payload = complete_query(
            required_patient_facts=["diagnosis", "renal function"]
        )
        with self.assertRaisesRegex(InvalidGateDecision, "partition"):
            parse_query_completeness(payload)

    def test_missing_facts_require_specific_clarification(self):
        with self.assertRaisesRegex(InvalidGateDecision, "clarification"):
            parse_query_completeness(
                incomplete_query(clarification_questions=[])
            )

    def test_evidence_gate_rejects_sufficiency_without_citation(self):
        with self.assertRaisesRegex(InvalidGateDecision, "cited ID"):
            parse_evidence_applicability(
                applicable_evidence(cited_evidence_ids=[])
            )

    def test_router_escalates_conflicting_or_outdated_evidence(self):
        query = parse_query_completeness(complete_query())
        conflict = parse_evidence_applicability(
            applicable_evidence(conflicting=True)
        )
        outdated = parse_evidence_applicability(
            applicable_evidence(outdated=True)
        )
        self.assertEqual(route_answerability(query, conflict).action, "escalate")
        self.assertEqual(route_answerability(query, outdated).action, "escalate")

    def test_false_premise_abstains_without_retrieval(self):
        query = parse_query_completeness(
            complete_query(
                known_patient_facts=[],
                required_patient_facts=[],
                query_complete=False,
                query_issue="false_premise",
            )
        )
        self.assertEqual(route_answerability(query).action, "abstain")


class AnswerabilityAgentV3Tests(unittest.TestCase):
    def _agent(self, query_gate, retriever, evidence_gate, generator):
        return build_answerability_agent(
            query_gate=query_gate,
            retriever=retriever,
            evidence_gate=evidence_gate,
            generator=generator,
        )

    def test_incomplete_query_clarifies_without_retrieval_or_generation(self):
        calls = []
        agent = self._agent(
            lambda _: incomplete_query(),
            lambda _: calls.append("retrieve"),
            lambda *_: calls.append("evidence"),
            lambda *_: calls.append("generate"),
        )
        result = agent.invoke(initial_answerability_state("Which treatment?"))
        self.assertEqual(result["action"], "clarify")
        self.assertIn("What diagnosis", result["final_answer"])
        self.assertEqual(calls, [])
        self.assertEqual(result["query_gate_calls"], 1)
        self.assertEqual(result["evidence_gate_calls"], 0)
        self.assertEqual(result["generation_calls"], 0)

    def test_answer_path_calls_each_dependency_once(self):
        calls = []

        def retrieve(query):
            calls.append(("retrieve", query))
            return RetrievalEvidence(("evidence text",), ("doc-1",), 0.01)

        agent = self._agent(
            lambda _: complete_query(),
            retrieve,
            lambda query, evidence: (
                calls.append(("evidence", query, evidence.evidence_ids))
                or applicable_evidence()
            ),
            lambda query, evidence: (
                calls.append(("generate", query, evidence.evidence_ids))
                or "Grounded answer [doc-1]"
            ),
        )
        result = agent.invoke(
            initial_answerability_state(
                "Question with answer options", retrieval_query="Question only"
            )
        )
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["final_answer"], "Grounded answer [doc-1]")
        self.assertEqual(calls[0], ("retrieve", "Question only"))
        self.assertEqual(calls[1][1], "Question with answer options")
        self.assertEqual(calls[2][1], "Question with answer options")
        self.assertEqual(result["query_gate_calls"], 1)
        self.assertEqual(result["evidence_gate_calls"], 1)
        self.assertEqual(result["generation_calls"], 1)

    def test_insufficient_evidence_abstains_without_generation(self):
        generated = []
        agent = self._agent(
            lambda _: complete_query(),
            lambda _: RetrievalEvidence(("generic text",), ("doc-1",)),
            lambda *_: applicable_evidence(
                sufficient=False,
                patient_applicable=False,
                cited_evidence_ids=[],
            ),
            lambda *_: generated.append(True) or "unsafe answer",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(generated, [])

    def test_conflicting_evidence_escalates_without_generation(self):
        generated = []
        agent = self._agent(
            lambda _: complete_query(),
            lambda _: RetrievalEvidence(("A", "B"), ("doc-1", "doc-2")),
            lambda *_: applicable_evidence(
                conflicting=True,
                cited_evidence_ids=["doc-1", "doc-2"],
            ),
            lambda *_: generated.append(True) or "unsafe answer",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "escalate")
        self.assertIn("conflicting", result["final_answer"])
        self.assertEqual(generated, [])

    def test_retrieval_failure_fails_closed_without_paid_evidence_gate(self):
        evidence_calls = []
        agent = self._agent(
            lambda _: complete_query(),
            lambda _: (_ for _ in ()).throw(RuntimeError("retriever offline")),
            lambda *_: evidence_calls.append(True) or applicable_evidence(),
            lambda *_: "unsafe answer",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(evidence_calls, [])
        self.assertEqual(result["evidence_gate_calls"], 0)
        self.assertTrue(result["evidence_gate"]["tool_failed"])

    def test_invalid_query_gate_fails_closed(self):
        retrieved = []
        agent = self._agent(
            lambda _: complete_query(query_complete=False),
            lambda _: retrieved.append(True),
            lambda *_: applicable_evidence(),
            lambda *_: "unsafe answer",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "query_gate_error")
        self.assertEqual(retrieved, [])
        self.assertIn("query_completeness_error", result["gate_errors"][0])

    def test_empty_generation_fails_closed(self):
        agent = self._agent(
            lambda _: complete_query(),
            lambda _: RetrievalEvidence(("evidence",), ("doc-1",)),
            lambda *_: applicable_evidence(),
            lambda *_: "",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "generation_error")
        self.assertIn("generation_error", result["gate_errors"][0])

    def test_evidence_gate_rejects_unretrieved_citation(self):
        agent = self._agent(
            lambda _: complete_query(),
            lambda _: RetrievalEvidence(("evidence",), ("doc-1",)),
            lambda *_: applicable_evidence(cited_evidence_ids=["doc-99"]),
            lambda *_: "should not run",
        )
        result = agent.invoke(initial_answerability_state("Question"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "evidence_gate_error")
        self.assertEqual(result["generation_calls"], 0)
        self.assertIn("not retrieved", result["gate_errors"][0])


if __name__ == "__main__":
    unittest.main()
