"""Zero-cost deterministic demo of the clinical handoff workflow."""

from __future__ import annotations

import json

from graphrag.agent.clinical_agent import (
    build_clinical_handoff_agent,
    initial_clinical_state,
    serializable_clinical_result,
)
from graphrag.agent.clinical_handoff import (
    ClinicalFact,
    ClinicalHandoff,
    DiagnosticDraft,
    EvidenceItem,
    InterviewDecision,
    InterviewUpdate,
    SafetyDecision,
)
from graphrag.agent.clinical_tools import ClinicalTools


def _interview(conversation, _previous_handoff):
    patient_turns = [turn for turn in conversation if turn.role == "patient"]
    facts = [
        ClinicalFact(
            fact_id="fact-chief",
            category="chief_complaint",
            statement="acute chest discomfort",
            status="present",
            source_turn_ids=("patient-0",),
        )
    ]
    if len(patient_turns) >= 2:
        facts.append(
            ClinicalFact(
                fact_id="fact-onset",
                category="timeline",
                statement="started two hours ago",
                status="present",
                source_turn_ids=("patient-1",),
            )
        )
    if len(patient_turns) >= 3:
        facts.append(
            ClinicalFact(
                fact_id="fact-radiation",
                category="symptom",
                statement="radiates to the left arm",
                status="present",
                source_turn_ids=("patient-2",),
            )
        )

    ready = len(patient_turns) >= 3
    missing = () if ready else ("onset", "radiation")[len(patient_turns) - 1 :]
    handoff = ClinicalHandoff(
        chief_complaint="acute chest discomfort",
        facts=tuple(facts),
        missing_information=missing,
        ready_for_diagnosis=ready,
    )
    if ready:
        decision = InterviewDecision("finalize", "Required fixture facts collected.")
    else:
        target = missing[0]
        question = (
            "When did the discomfort begin?"
            if target == "onset"
            else "Does the discomfort spread anywhere else?"
        )
        decision = InterviewDecision(
            action="ask",
            reason="This fact can change urgency and the differential.",
            question=question,
            target_information=target,
        )
    return InterviewUpdate(handoff=handoff, decision=decision)


def _ask_patient(question, _conversation):
    if "begin" in question:
        return "It started about two hours ago."
    return "It spreads to my left arm."


def _retrieve(_handoff):
    return (
        EvidenceItem(
            evidence_id="textbook-001",
            text=(
                "Acute chest discomfort with arm radiation warrants urgent "
                "in-person assessment for time-sensitive cardiac causes."
            ),
            source="offline-demo-textbook-fixture",
        ),
    )


def _diagnose(_packet):
    return DiagnosticDraft(
        answer=(
            "This simulated case warrants urgent in-person cardiac evaluation "
            "[fact-chief, fact-onset, fact-radiation, textbook-001]."
        ),
        differential=("time-sensitive cardiac cause",),
        recommendations=("urgent in-person evaluation",),
        cited_fact_ids=("fact-chief", "fact-onset", "fact-radiation"),
        cited_evidence_ids=("textbook-001",),
        confidence=0.75,
    )


def main() -> None:
    agent = build_clinical_handoff_agent(
        interview=_interview,
        tools=ClinicalTools(
            ask_patient=_ask_patient,
            retrieve_evidence=_retrieve,
        ),
        diagnose=_diagnose,
        validate_safety=lambda *_: SafetyDecision(
            "approve", "Fixture answer is grounded and recommends evaluation."
        ),
    )
    result = agent.invoke(initial_clinical_state("I have sudden chest discomfort."))
    print(json.dumps(serializable_clinical_result(result), indent=2))


if __name__ == "__main__":
    main()
