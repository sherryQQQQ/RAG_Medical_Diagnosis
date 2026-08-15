import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from graphrag.agent.clinical_handoff import ConversationTurn
from graphrag.eval.mediq_handoff_benchmark import (
    DIAGNOSIS_SCHEMA,
    DIAGNOSTIC_CONDITIONS,
    INTERVIEW_SCHEMA,
    CheckpointedGeminiProvider,
    DeterministicFactPatient,
    MediQCase,
    ProviderResponse,
    _diagnosis_prompt,
    _case_run,
    dry_run_plan,
    load_cases,
    load_spec,
    parse_diagnosis,
    parse_interview_update,
)
from graphrag.eval.mediq_handoff_data import ConceptAwareFactPatient


class Snippet:
    def __init__(self):
        self.snippet_id = "doc-1"
        self.title = "Fixture textbook"
        self.content = "Persistent fever requires evaluation of the clinical context."


class Retriever:
    def retrieve(self, query, k=8):
        self.query = query
        self.k = k
        return [Snippet()]


def fixture_case():
    return MediQCase(
        case_id="mediq-fixture",
        source_id=1,
        specialty="Pediatrics",
        question="What is the most likely diagnosis?",
        initial_info="A child has fever.",
        context=("A child has fever.", "The fever has lasted three days."),
        facts=("A child has fever.", "The fever has lasted three days."),
        options={"A": "Diagnosis one", "B": "Diagnosis two"},
        answer_choice="A",
    )


class MediQSourceTests(unittest.TestCase):
    def test_committed_spec_freezes_five_case_budget(self):
        spec = load_spec()
        self.assertEqual(len(spec["selected_source_ids"]), 5)
        self.assertEqual(spec["diagnostic_conditions"], list(DIAGNOSTIC_CONDITIONS))
        plan = dry_run_plan([fixture_case()] * 5, spec, "gemini-2.5-flash")
        self.assertEqual(plan["max_provider_calls"], 35)
        self.assertEqual(plan["patient_model_calls"], 0)
        self.assertEqual(plan["judge_model_calls"], 0)
        self.assertLess(plan["theoretical_cost_bound_usd"], 0.25)

    def test_selection_recomputes_seeded_specialty_sample(self):
        base = load_spec()
        rows = []
        for source_id, specialty in zip(
            base["selected_source_ids"], base["specialties"]
        ):
            rows.append(
                {
                    "id": source_id,
                    "question": "Fixture question",
                    "context": ["Initial information", "Additional information"],
                    "facts": ["1. Initial information", "2. Additional information"],
                    "options": {"A": "One", "B": "Two"},
                    "answer_idx": "A",
                    "patient": {"gpt_specialty": specialty},
                }
            )
        raw = "".join(json.dumps(row) + "\n" for row in rows)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(raw, encoding="utf-8")
            spec = dict(base)
            spec["source_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            cases = load_cases(source, spec)
        self.assertEqual(
            [case.source_id for case in cases], base["selected_source_ids"]
        )


class HandoffParsingTests(unittest.TestCase):
    def test_interview_rejects_an_invented_patient_turn(self):
        raw = json.dumps(
            {
                "chief_complaint": "fever",
                "facts": [
                    {
                        "fact_id": "f-1",
                        "category": "symptom",
                        "statement": "fever",
                        "status": "present",
                        "source_turn_ids": ["patient-99"],
                    }
                ],
                "missing_information": [],
                "contradictions": [],
                "ready_for_diagnosis": True,
                "action": "finalize",
                "reason": "Enough information",
                "question": "",
                "target_information": "",
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown patient turn"):
            parse_interview_update(
                raw, (ConversationTurn("patient-0", "patient", "I have fever"),)
            )

    def test_interview_normalizes_inconsistent_readiness_downward(self):
        raw = json.dumps(
            {
                "chief_complaint": "fever",
                "facts": [
                    {
                        "fact_id": "f-1",
                        "category": "symptom",
                        "statement": "fever",
                        "status": "present",
                        "source_turn_ids": ["patient-0"],
                    }
                ],
                "missing_information": ["duration"],
                "contradictions": [],
                "ready_for_diagnosis": True,
                "action": "finalize",
                "reason": "Provider claimed readiness.",
                "question": "",
                "target_information": "",
            }
        )
        update = parse_interview_update(
            raw, (ConversationTurn("patient-0", "patient", "I have fever"),)
        )
        self.assertFalse(update.handoff.ready_for_diagnosis)
        self.assertEqual(update.handoff.missing_information, ("duration",))

    def test_diagnosis_rejects_unretrieved_citation(self):
        raw = json.dumps(
            {
                "answer_choice": "A",
                "answer": "Fixture",
                "differential": [],
                "recommendations": [],
                "cited_patient_ids": ["f-1"],
                "cited_evidence_ids": ["invented-doc"],
                "confidence": 0.5,
            }
        )
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            parse_diagnosis(
                raw,
                options={"A": "One", "B": "Two"},
                allowed_patient_ids={"f-1"},
                allowed_evidence_ids={"doc-1"},
            )

    def test_deterministic_patient_uses_no_model_and_reveals_matching_fact(self):
        patient = DeterministicFactPatient(fixture_case())
        answer = patient(
            "How long has the fever lasted?",
            (ConversationTurn("patient-0", "patient", "A child has fever."),),
        )
        self.assertIn("three days", answer)
        self.assertIn("source-fact-2", patient.revealed)

    def test_concept_patient_maps_generic_history_to_diabetes(self):
        case = MediQCase(
            case_id="mediq-history",
            source_id=2,
            specialty="Internal Medicine",
            question="What is the diagnosis?",
            initial_info="The patient reports irritation.",
            context=("The patient reports irritation.",),
            facts=("The patient has type 2 diabetes mellitus.",),
            options={"A": "One", "B": "Two"},
            answer_choice="A",
        )
        lexical = DeterministicFactPatient(case)
        concept = ConceptAwareFactPatient(case)
        conversation = (
            ConversationTurn("patient-0", "patient", case.initial_info),
        )

        lexical_answer = lexical("Do you have any relevant medical history?", conversation)
        concept_answer = concept("Do you have any relevant medical history?", conversation)

        self.assertIn("cannot answer", lexical_answer)
        self.assertIn("diabetes", concept_answer)
        self.assertEqual(concept.matcher_version, "clinical-concept-v2")

    def test_full_transcript_includes_questions_but_cites_only_patient_turns(self):
        from graphrag.agent.clinical_handoff import (
            ClinicalFact,
            ClinicalHandoff,
            DiagnosticPacket,
            EvidenceItem,
        )

        conversation = (
            ConversationTurn("patient-0", "patient", "I have fever."),
            ConversationTurn("agent-1", "agent", "How long has it lasted?"),
            ConversationTurn("patient-1", "patient", "Three days."),
        )
        packet = DiagnosticPacket(
            handoff=ClinicalHandoff(
                chief_complaint="fever",
                facts=(
                    ClinicalFact("f-1", "symptom", "fever", "present", ("patient-0",)),
                ),
            ),
            evidence=(EvidenceItem("doc-1", "fixture", "textbook"),),
            source_turns=(conversation[0],),
        )

        prompt, allowed_ids = _diagnosis_prompt(
            fixture_case(), packet, "full-transcript", conversation
        )

        self.assertIn("How long has it lasted?", prompt)
        self.assertEqual(allowed_ids, {"patient-0", "patient-1"})
        self.assertNotIn("'agent-1'", str(sorted(allowed_ids)))


class PilotExecutionTests(unittest.TestCase):
    def test_checkpoint_reuses_saved_provider_response_without_api_call(self):
        prompt = "fixture prompt"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "model": "gemini-2.5-flash",
                        "dataset_fingerprint": "fixture",
                        "calls": {
                            "call-1": {
                                "stage": "interview",
                                "prompt_sha256": hashlib.sha256(
                                    prompt.encode()
                                ).hexdigest(),
                                "raw_output": "{}",
                                "latency_s": 1.0,
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "total_tokens": 12,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            provider = CheckpointedGeminiProvider(
                model="gemini-2.5-flash",
                checkpoint_path=path,
                fingerprint="fixture",
            )
            response = provider("call-1", "interview", prompt, INTERVIEW_SCHEMA)
        self.assertTrue(response.reused)
        self.assertEqual(response.total_tokens, 12)

    def test_reuse_only_provider_fails_before_a_missing_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = CheckpointedGeminiProvider(
                model="gemini-2.5-flash",
                checkpoint_path=Path(directory) / "checkpoint.json",
                fingerprint="fixture",
                allow_new_calls=False,
            )
            with self.assertRaisesRegex(RuntimeError, "Reuse-only"):
                provider("missing", "interview", "prompt", INTERVIEW_SCHEMA)

    def test_offline_runner_completes_three_diagnostic_conditions(self):
        calls = []

        def provider(call_id, stage, prompt, schema):
            calls.append((call_id, stage, schema))
            if stage == "interview":
                if call_id.endswith(":0"):
                    payload = {
                        "chief_complaint": "fever",
                        "facts": [
                            {
                                "fact_id": "f-chief",
                                "category": "symptom",
                                "statement": "child has fever",
                                "status": "present",
                                "source_turn_ids": ["patient-0"],
                            }
                        ],
                        "missing_information": ["duration"],
                        "contradictions": [],
                        "ready_for_diagnosis": False,
                        "action": "ask",
                        "reason": "Duration is discriminative.",
                        "question": "How long has the fever lasted?",
                        "target_information": "duration",
                    }
                else:
                    payload = {
                        "chief_complaint": "fever",
                        "facts": [
                            {
                                "fact_id": "f-chief",
                                "category": "symptom",
                                "statement": "child has fever",
                                "status": "present",
                                "source_turn_ids": ["patient-0"],
                            },
                            {
                                "fact_id": "f-duration",
                                "category": "timeline",
                                "statement": "fever lasted three days",
                                "status": "present",
                                "source_turn_ids": ["patient-1"],
                            },
                        ],
                        "missing_information": [],
                        "contradictions": [],
                        "ready_for_diagnosis": True,
                        "action": "finalize",
                        "reason": "Enough information.",
                        "question": "",
                        "target_information": "",
                    }
            else:
                if "raw-conversation" in stage or "full-transcript" in stage:
                    patient_ids = ["patient-0"]
                else:
                    patient_ids = ["f-chief"]
                payload = {
                    "answer_choice": "A",
                    "answer": "Diagnosis one",
                    "differential": ["Diagnosis one"],
                    "recommendations": ["Fixture evaluation"],
                    "cited_patient_ids": patient_ids,
                    "cited_evidence_ids": ["doc-1"],
                    "confidence": 0.7,
                }
            return ProviderResponse(
                raw_output=json.dumps(payload),
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                latency_s=0.1,
                reused=False,
            )

        retriever = Retriever()
        result = _case_run(
            fixture_case(),
            provider=provider,
            retriever=retriever,
            max_questions=3,
            top_k=8,
        )
        self.assertEqual(result["questions_asked"], 1)
        self.assertEqual(set(result["diagnoses"]), set(DIAGNOSTIC_CONDITIONS))
        self.assertTrue(all(item["correct"] for item in result["diagnoses"].values()))
        self.assertEqual(len(calls), 5)
        self.assertEqual(retriever.k, 8)
        self.assertTrue(result["retrieval"]["options_excluded"])

        holdout = _case_run(
            fixture_case(),
            provider=provider,
            retriever=Retriever(),
            max_questions=3,
            top_k=8,
            diagnostic_conditions=(
                "full-transcript",
                "structured-handoff",
                "handoff-plus-sources",
            ),
            patient_factory=ConceptAwareFactPatient,
        )
        self.assertEqual(
            set(holdout["diagnoses"]),
            {"full-transcript", "structured-handoff", "handoff-plus-sources"},
        )
        self.assertTrue(
            all(item["correct"] for item in holdout["diagnoses"].values())
        )


if __name__ == "__main__":
    unittest.main()
