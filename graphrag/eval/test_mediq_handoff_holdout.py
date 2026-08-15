import hashlib
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from graphrag.agent.clinical_handoff import ClinicalHandoff, ConversationTurn
from graphrag.eval.mediq_handoff_data import (
    CoverageAwareQuestionRefiner,
    MediQCase,
    clinical_concepts,
)
from graphrag.eval.mediq_handoff_holdout import (
    HOLDOUT_DIAGNOSTIC_CONDITIONS,
    load_holdout_cases,
    load_holdout_spec,
    holdout_dry_run_plan,
)


def fixture_case(source_id: int = 1) -> MediQCase:
    return MediQCase(
        case_id=f"mediq-{source_id}",
        source_id=source_id,
        specialty="Pediatrics",
        question="What is the most likely diagnosis?",
        initial_info="A child has fever.",
        context=("A child has fever.",),
        facts=("A child has fever.",),
        options={"A": "Diagnosis one", "B": "Diagnosis two"},
        answer_choice="A",
    )


class ConceptMappingTests(unittest.TestCase):
    def test_vital_units_and_trauma_terms_map_without_substring_leakage(self):
        self.assertIn("vital_sign", clinical_concepts("BP 90/60 and SpO2 92%"))
        self.assertIn("trauma", clinical_concepts("The patient has a laceration"))
        self.assertNotIn("trauma", clinical_concepts("Following the visit"))

    def test_question_refiner_replaces_repeated_concept_with_uncovered_history(self):
        refiner = CoverageAwareQuestionRefiner()
        conversation = (
            ConversationTurn("patient-0", "patient", "I have pain and discharge."),
            ConversationTurn("agent-1", "agent", "Are you sexually active?"),
            ConversationTurn("patient-1", "patient", "Yes."),
            ConversationTurn("agent-2", "agent", "When did this start?"),
            ConversationTurn("patient-2", "patient", "Last week."),
        )

        refined = refiner(
            "Do you use an IUD?",
            conversation,
            ClinicalHandoff(chief_complaint="pain", facts=()),
        )

        self.assertIn("past medical conditions", refined)
        self.assertNotIn("diabetes", refined)

    def test_reproductive_wording_maps_to_one_concept(self):
        for wording in (
            "Are you sexually active?",
            "Was there unprotected sexual intercourse?",
            "Do you douche frequently?",
        ):
            self.assertIn("reproductive", clinical_concepts(wording))

    def test_refiner_keeps_a_new_vital_sign_question_after_symptoms(self):
        refiner = CoverageAwareQuestionRefiner()
        conversation = (
            ConversationTurn("patient-0", "patient", "I have a fever."),
            ConversationTurn("agent-1", "agent", "What other symptoms do you have?"),
            ConversationTurn("patient-1", "patient", "I feel weak."),
        )
        question = "What are your current vital signs?"

        refined = refiner(
            question,
            conversation,
            ClinicalHandoff(chief_complaint="fever", facts=()),
        )

        self.assertEqual(refined, question)


class HoldoutContractTests(unittest.TestCase):
    def test_committed_spec_freezes_balanced_unseen_holdout(self):
        spec = load_holdout_spec()
        self.assertEqual(len(spec["selected_source_ids"]), 30)
        self.assertFalse(
            set(spec["selected_source_ids"]) & set(spec["excluded_source_ids"])
        )
        self.assertEqual(
            spec["diagnostic_conditions"], list(HOLDOUT_DIAGNOSTIC_CONDITIONS)
        )

    def test_seeded_selection_is_recomputed_from_source_metadata(self):
        base = load_holdout_spec()
        rows = []
        next_id = 10_000
        candidates_by_specialty = {}
        for specialty in base["specialties"]:
            candidates = []
            for _ in range(7):
                row = {
                    "id": next_id,
                    "question": "Fixture question",
                    "context": ["Initial information"],
                    "facts": ["1. Initial information"],
                    "options": {"A": "One", "B": "Two"},
                    "answer_idx": "A",
                    "patient": {"gpt_specialty": specialty},
                }
                rows.append(row)
                candidates.append(row)
                next_id += 1
            candidates_by_specialty[specialty] = candidates
        rng = random.Random(int(base["seed"]))
        selected_ids = []
        for specialty in base["specialties"]:
            selected_ids.extend(
                int(row["id"])
                for row in rng.sample(
                    candidates_by_specialty[specialty],
                    int(base["cases_per_specialty"]),
                )
            )
        raw = "".join(json.dumps(row) + "\n" for row in rows)
        spec = dict(base)
        spec["source_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        spec["selected_source_ids"] = selected_ids
        spec["excluded_source_ids"] = []

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(raw, encoding="utf-8")
            cases = load_holdout_cases(source, spec)

        self.assertEqual([case.source_id for case in cases], selected_ids)
        self.assertEqual(Counter(case.specialty for case in cases), Counter({
            specialty: 6 for specialty in base["specialties"]
        }))

    def test_dry_run_has_hard_call_and_cost_bounds(self):
        spec = load_holdout_spec()
        cases = [fixture_case(source_id) for source_id in range(30)]
        plan = holdout_dry_run_plan(cases, spec, "gemini-2.5-flash")

        self.assertEqual(plan["max_provider_calls"], 210)
        self.assertEqual(plan["patient_model_calls"], 0)
        self.assertEqual(plan["judge_model_calls"], 0)
        self.assertLess(plan["theoretical_cost_bound_usd"], 2.0)
        self.assertAlmostEqual(
            plan["expected_cost_extrapolated_from_stage5n_usd"], 0.3010086
        )
        self.assertTrue(plan["requires_explicit_execute_flag"])


if __name__ == "__main__":
    unittest.main()
