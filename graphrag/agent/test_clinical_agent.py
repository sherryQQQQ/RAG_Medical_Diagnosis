import unittest

from graphrag.agent.clinical_agent import (
    build_clinical_handoff_agent,
    initial_clinical_state,
    serializable_clinical_result,
)
from graphrag.agent.clinical_handoff import (
    ClinicalFact,
    ClinicalHandoff,
    ConversationTurn,
    DiagnosticDraft,
    DiagnosticPacket,
    EvidenceItem,
    InterviewDecision,
    InterviewUpdate,
    SafetyDecision,
)
from graphrag.agent.clinical_tools import ClinicalTools


def fact(fact_id, statement, source, *, category="symptom", status="present"):
    return ClinicalFact(
        fact_id=fact_id,
        category=category,
        statement=statement,
        status=status,
        source_turn_ids=(source,),
    )


def ready_handoff(_conversation=None):
    return ClinicalHandoff(
        chief_complaint="chest pain",
        facts=(fact("f-chest-pain", "acute chest pain", "patient-0"),),
        ready_for_diagnosis=True,
    )


def evidence_tool(_):
    return (
        EvidenceItem(
            evidence_id="doc-1",
            text="Acute radiating chest pain requires urgent cardiac evaluation.",
            source="medical-textbook",
        ),
    )


def grounded_draft(_):
    return DiagnosticDraft(
        answer="Urgent cardiac evaluation is warranted [f-chest-pain, doc-1].",
        differential=("acute coronary syndrome",),
        recommendations=("seek urgent in-person evaluation",),
        cited_fact_ids=("f-chest-pain",),
        cited_evidence_ids=("doc-1",),
        confidence=0.72,
    )


class ClinicalHandoffContractTests(unittest.TestCase):
    def test_fact_requires_patient_turn_provenance(self):
        with self.assertRaisesRegex(ValueError, "patient-turn source"):
            fact("f-1", "fever", "")

    def test_ready_handoff_rejects_unresolved_contradictions(self):
        with self.assertRaisesRegex(ValueError, "contradictions"):
            ClinicalHandoff(
                chief_complaint="fever",
                facts=(fact("f-1", "fever", "patient-0"),),
                contradictions=("fever both present and absent",),
                ready_for_diagnosis=True,
            )

    def test_question_budget_is_hard_capped_at_three(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            initial_clinical_state("I feel ill", max_questions=4)

    def test_ready_handoff_cannot_retain_missing_information(self):
        with self.assertRaisesRegex(ValueError, "missing information"):
            ClinicalHandoff(
                chief_complaint="fever",
                facts=(fact("f-1", "fever", "patient-0"),),
                missing_information=("duration",),
                ready_for_diagnosis=True,
            )

    def test_diagnostic_packet_rejects_incomplete_provenance(self):
        handoff = ClinicalHandoff(
            chief_complaint="fever",
            facts=(
                fact("f-1", "fever", "patient-0"),
                fact("f-2", "two days", "patient-1", category="timeline"),
            ),
            ready_for_diagnosis=True,
        )
        with self.assertRaisesRegex(ValueError, "missing cited patient turns"):
            DiagnosticPacket(
                handoff=handoff,
                evidence=evidence_tool(handoff),
                source_turns=(
                    ConversationTurn("patient-0", "patient", "I have a fever."),
                ),
            )


class ClinicalHandoffAgentTests(unittest.TestCase):
    def _agent(
        self,
        *,
        extract=ready_handoff,
        plan=lambda *_: InterviewDecision("finalize", "Enough information."),
        patient=lambda *_: "fixture response",
        retrieve=evidence_tool,
        diagnose=grounded_draft,
        safety=lambda *_: SafetyDecision("approve", "Grounded and safe."),
        refine=None,
    ):
        def interviewer(conversation, _previous_handoff):
            handoff = extract(conversation) if callable(extract) else extract
            return InterviewUpdate(
                handoff=handoff,
                decision=plan(handoff, conversation),
            )

        return build_clinical_handoff_agent(
            interview=interviewer,
            tools=ClinicalTools(
                ask_patient=patient,
                retrieve_evidence=retrieve,
                refine_question=refine,
            ),
            diagnose=diagnose,
            validate_safety=safety,
        )

    def test_ready_case_retrieves_diagnoses_and_passes_safety_gate(self):
        packets = []

        def diagnose(packet):
            packets.append(packet)
            return grounded_draft(packet)

        result = self._agent(diagnose=diagnose).invoke(
            initial_clinical_state("I have sudden chest pain")
        )

        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["questions_asked"], 0)
        self.assertEqual(result["retrieval_tool_calls"], 1)
        self.assertEqual(result["diagnostic_calls"], 1)
        self.assertEqual(result["safety_calls"], 1)
        self.assertEqual(
            [turn.turn_id for turn in packets[0].source_turns], ["patient-0"]
        )

    def test_two_question_interview_updates_handoff_before_diagnosis(self):
        patient_answers = iter(("It started two hours ago.", "It radiates to my arm."))
        questions = []

        def extract(conversation):
            patient_turns = [turn for turn in conversation if turn.role == "patient"]
            facts = [fact("f-chief", "chest pain", "patient-0")]
            if len(patient_turns) >= 2:
                facts.append(fact("f-onset", "started two hours ago", "patient-1"))
            if len(patient_turns) >= 3:
                facts.append(fact("f-radiation", "radiates to arm", "patient-2"))
            return ClinicalHandoff(
                chief_complaint="chest pain",
                facts=tuple(facts),
                missing_information=(
                    ()
                    if len(patient_turns) >= 3
                    else ("onset", "radiation")[len(patient_turns) - 1 :]
                ),
                ready_for_diagnosis=len(patient_turns) >= 3,
            )

        def plan(handoff, _):
            if handoff.ready_for_diagnosis:
                return InterviewDecision("finalize", "Enough information.")
            target = handoff.missing_information[0]
            return InterviewDecision(
                action="ask",
                question=f"Please provide {target}.",
                target_information=target,
                reason="This fact could change the differential.",
            )

        def patient(question, _):
            questions.append(question)
            return next(patient_answers)

        result = self._agent(
            extract=extract,
            plan=plan,
            patient=patient,
            diagnose=lambda _: DiagnosticDraft(
                answer="Urgent evaluation is warranted [f-chief, f-onset, doc-1].",
                differential=("acute coronary syndrome",),
                recommendations=("urgent evaluation",),
                cited_fact_ids=("f-chief", "f-onset"),
                cited_evidence_ids=("doc-1",),
                confidence=0.7,
            ),
        ).invoke(initial_clinical_state("I have chest pain"))

        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["questions_asked"], 2)
        self.assertEqual(result["patient_tool_calls"], 2)
        self.assertEqual(result["patient_tool_successes"], 2)
        self.assertEqual(result["interview_calls"], 3)
        self.assertEqual(len(questions), 2)

    def test_question_budget_exhaustion_abstains_before_retrieval(self):
        calls = []

        def incomplete(conversation):
            return ClinicalHandoff(
                chief_complaint="dizziness",
                facts=(fact("f-1", "dizziness", "patient-0"),),
                missing_information=("onset",),
                ready_for_diagnosis=False,
            )

        result = self._agent(
            extract=incomplete,
            plan=lambda *_: InterviewDecision(
                "ask", "Need onset.", "When did it start?", "onset"
            ),
            patient=lambda *_: "I do not know.",
            retrieve=lambda _: calls.append("retrieve") or evidence_tool(None),
            diagnose=lambda _: calls.append("diagnose") or grounded_draft(None),
        ).invoke(initial_clinical_state("I feel dizzy", max_questions=3))

        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "question_budget_exhausted")
        self.assertEqual(result["questions_asked"], 3)
        self.assertEqual(result["interview_calls"], 4)
        self.assertEqual(calls, [])

    def test_benchmark_mode_forces_diagnosis_at_question_budget(self):
        def incomplete(_conversation):
            return ClinicalHandoff(
                chief_complaint="dizziness",
                facts=(fact("f-1", "dizziness", "patient-0"),),
                missing_information=("onset",),
                ready_for_diagnosis=False,
            )

        result = self._agent(
            extract=incomplete,
            plan=lambda *_: InterviewDecision(
                "ask", "Need onset.", "When did it start?", "onset"
            ),
            patient=lambda *_: "I do not know.",
            diagnose=lambda _: DiagnosticDraft(
                answer="A best-effort benchmark answer [f-1, doc-1].",
                differential=("fixture",),
                recommendations=("fixture",),
                cited_fact_ids=("f-1",),
                cited_evidence_ids=("doc-1",),
                confidence=0.2,
            ),
        ).invoke(
            initial_clinical_state(
                "I feel dizzy",
                max_questions=3,
                force_diagnosis_on_budget=True,
            )
        )

        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["questions_asked"], 3)
        self.assertEqual(result["interview_calls"], 4)
        self.assertEqual(
            result["interview_decision"].action,
            "finalize",
        )

    def test_question_refinement_tool_changes_the_recorded_question(self):
        asked = []

        def extract(conversation):
            if len([turn for turn in conversation if turn.role == "patient"]) == 1:
                return ClinicalHandoff(
                    chief_complaint="fever",
                    facts=(fact("f-1", "fever", "patient-0"),),
                    missing_information=("duration",),
                )
            return ClinicalHandoff(
                chief_complaint="fever",
                facts=(fact("f-1", "fever", "patient-0"),),
                ready_for_diagnosis=True,
            )

        def patient(question, _conversation):
            asked.append(question)
            return "Three days."

        result = self._agent(
            extract=extract,
            plan=lambda handoff, *_: (
                InterviewDecision("finalize", "Enough information.")
                if handoff.ready_for_diagnosis
                else InterviewDecision("ask", "Need duration.", "Original?", "duration")
            ),
            patient=patient,
            refine=lambda *_: "How long has the fever lasted?",
            diagnose=lambda _: DiagnosticDraft(
                answer="Fixture answer [f-1, doc-1].",
                differential=("fixture",),
                recommendations=("fixture",),
                cited_fact_ids=("f-1",),
                cited_evidence_ids=("doc-1",),
                confidence=0.5,
            ),
        ).invoke(initial_clinical_state("I have fever"))

        self.assertEqual(asked, ["How long has the fever lasted?"])
        self.assertEqual(result["conversation"][1].content, asked[0])
        self.assertIn("question_refined", result["trace"])

    def test_incomplete_finalize_requires_explicit_benchmark_override(self):
        incomplete = ClinicalHandoff(
            chief_complaint="dizziness",
            facts=(fact("f-1", "dizziness", "patient-0"),),
            missing_information=("onset",),
            ready_for_diagnosis=False,
        )
        diagnosis = lambda _: DiagnosticDraft(
            answer="A benchmark answer [f-1, doc-1].",
            differential=("fixture",),
            recommendations=("fixture",),
            cited_fact_ids=("f-1",),
            cited_evidence_ids=("doc-1",),
            confidence=0.2,
        )
        agent = self._agent(
            extract=incomplete,
            plan=lambda *_: InterviewDecision("finalize", "Best effort."),
            diagnose=diagnosis,
        )

        safe = agent.invoke(initial_clinical_state("I feel dizzy"))
        benchmark = agent.invoke(
            initial_clinical_state(
                "I feel dizzy", allow_incomplete_finalize=True
            )
        )

        self.assertEqual(safe["status"], "incomplete_handoff")
        self.assertEqual(benchmark["action"], "answer")

    def test_handoff_cannot_cite_a_nonexistent_patient_turn(self):
        def invalid(_):
            return ClinicalHandoff(
                chief_complaint="fever",
                facts=(fact("f-1", "fever", "patient-99"),),
                ready_for_diagnosis=True,
            )

        result = self._agent(extract=invalid).invoke(
            initial_clinical_state("I have a fever")
        )
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "interview_error")
        self.assertEqual(result["retrieval_tool_calls"], 0)

    def test_patient_tool_failure_fails_closed(self):
        incomplete = ClinicalHandoff(
            chief_complaint="pain",
            facts=(fact("f-1", "pain", "patient-0"),),
            missing_information=("location",),
            ready_for_diagnosis=False,
        )
        result = self._agent(
            extract=incomplete,
            plan=lambda *_: InterviewDecision(
                "ask", "Need location.", "Where is the pain?", "location"
            ),
            patient=lambda *_: (_ for _ in ()).throw(RuntimeError("simulator down")),
        ).invoke(initial_clinical_state("I have pain"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "patient_tool_error")
        self.assertEqual(result["patient_tool_calls"], 1)
        self.assertEqual(result["patient_tool_successes"], 0)

    def test_retrieval_failure_fails_closed_without_diagnosis(self):
        diagnosed = []
        result = self._agent(
            retrieve=lambda _: (_ for _ in ()).throw(RuntimeError("index offline")),
            diagnose=lambda _: diagnosed.append(True) or grounded_draft(None),
        ).invoke(initial_clinical_state("I have chest pain"))
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "retrieval_tool_error")
        self.assertEqual(result["retrieval_tool_calls"], 1)
        self.assertEqual(result["retrieval_tool_successes"], 0)
        self.assertEqual(diagnosed, [])

    def test_diagnostic_hallucinated_citation_is_rejected(self):
        def bad_draft(_):
            return DiagnosticDraft(
                answer="Unsupported answer.",
                differential=("fixture",),
                recommendations=("fixture",),
                cited_fact_ids=("invented-fact",),
                cited_evidence_ids=("doc-1",),
                confidence=0.9,
            )

        result = self._agent(diagnose=bad_draft).invoke(
            initial_clinical_state("I have chest pain")
        )
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["status"], "diagnostic_error")
        self.assertEqual(result["safety_calls"], 0)

    def test_safety_gate_can_escalate_without_exposing_draft(self):
        result = self._agent(
            safety=lambda *_: SafetyDecision(
                "escalate", "Possible emergency requires clinician review."
            )
        ).invoke(initial_clinical_state("I have sudden chest pain"))
        self.assertEqual(result["action"], "escalate")
        self.assertNotIn("acute coronary syndrome", result["final_answer"])

    def test_source_turns_can_be_disabled_for_handoff_ablation(self):
        packets = []

        def diagnose(packet):
            packets.append(packet)
            return grounded_draft(packet)

        result = self._agent(diagnose=diagnose).invoke(
            initial_clinical_state(
                "I have sudden chest pain", include_source_turns=False
            )
        )
        self.assertEqual(result["action"], "answer")
        self.assertEqual(packets[0].source_turns, ())

    def test_result_trace_is_json_serializable_shape(self):
        result = self._agent().invoke(
            initial_clinical_state("I have sudden chest pain")
        )
        serializable = serializable_clinical_result(result)
        self.assertIsInstance(serializable["conversation"][0], dict)
        self.assertEqual(
            serializable["diagnostic_packet"]["evidence"][0]["evidence_id"],
            "doc-1",
        )


if __name__ == "__main__":
    unittest.main()
