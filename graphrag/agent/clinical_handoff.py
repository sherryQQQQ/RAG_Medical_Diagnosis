"""Typed contracts for the interactive clinical handoff Agent.

The interview model and the diagnostic model communicate through these
contracts instead of sharing hidden chat state.  Every patient fact points back
to one or more patient turns so downstream decisions can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass


HANDOFF_SCHEMA_VERSION = 1
FACT_STATUSES = {"present", "absent", "unknown"}
INTERVIEW_ACTIONS = {"ask", "finalize"}
SAFETY_ACTIONS = {"approve", "abstain", "escalate"}


def _required_text(value: str, field: str) -> str:
    text = " ".join(str(value).split())
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _unique_text(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        text = _required_text(value, field)
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    role: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _required_text(self.turn_id, "turn_id"))
        if self.role not in {"patient", "agent"}:
            raise ValueError("role must be patient or agent")
        object.__setattr__(self, "content", _required_text(self.content, "content"))


@dataclass(frozen=True)
class ClinicalFact:
    fact_id: str
    category: str
    statement: str
    status: str
    source_turn_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _required_text(self.fact_id, "fact_id"))
        object.__setattr__(self, "category", _required_text(self.category, "category"))
        object.__setattr__(
            self, "statement", _required_text(self.statement, "statement")
        )
        if self.status not in FACT_STATUSES:
            raise ValueError(f"Unsupported fact status: {self.status}")
        try:
            sources = _unique_text(self.source_turn_ids, "source_turn_ids")
        except ValueError as error:
            raise ValueError(
                "Each clinical fact requires a patient-turn source"
            ) from error
        if not sources:
            raise ValueError("Each clinical fact requires a patient-turn source")
        object.__setattr__(self, "source_turn_ids", sources)


@dataclass(frozen=True)
class ClinicalHandoff:
    chief_complaint: str
    facts: tuple[ClinicalFact, ...]
    missing_information: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    ready_for_diagnosis: bool = False
    schema_version: int = HANDOFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HANDOFF_SCHEMA_VERSION:
            raise ValueError("Unsupported clinical-handoff schema version")
        object.__setattr__(
            self,
            "chief_complaint",
            _required_text(self.chief_complaint, "chief_complaint"),
        )
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Clinical fact IDs must be unique")
        object.__setattr__(
            self,
            "missing_information",
            _unique_text(self.missing_information, "missing_information"),
        )
        object.__setattr__(
            self,
            "contradictions",
            _unique_text(self.contradictions, "contradictions"),
        )
        if self.ready_for_diagnosis and self.missing_information:
            raise ValueError("A ready handoff cannot retain missing information")
        if self.ready_for_diagnosis and self.contradictions:
            raise ValueError("A handoff with unresolved contradictions is not ready")


@dataclass(frozen=True)
class InterviewDecision:
    action: str
    reason: str
    question: str = ""
    target_information: str = ""

    def __post_init__(self) -> None:
        if self.action not in INTERVIEW_ACTIONS:
            raise ValueError(f"Unsupported interview action: {self.action}")
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        if self.action == "ask":
            object.__setattr__(
                self, "question", _required_text(self.question, "question")
            )
            object.__setattr__(
                self,
                "target_information",
                _required_text(self.target_information, "target_information"),
            )
        elif self.question or self.target_information:
            raise ValueError("A finalize decision cannot contain another question")


@dataclass(frozen=True)
class InterviewUpdate:
    """One structured interviewer output: state update plus next action."""

    handoff: ClinicalHandoff
    decision: InterviewDecision


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    text: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _required_text(self.evidence_id, "evidence_id")
        )
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(self, "source", _required_text(self.source, "source"))


@dataclass(frozen=True)
class DiagnosticPacket:
    """The only input visible to the fresh diagnostic model."""

    handoff: ClinicalHandoff
    evidence: tuple[EvidenceItem, ...]
    source_turns: tuple[ConversationTurn, ...] = ()

    def __post_init__(self) -> None:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique")
        source_ids = [turn.turn_id for turn in self.source_turns]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source turn IDs must be unique")
        if any(turn.role != "patient" for turn in self.source_turns):
            raise ValueError("Diagnostic provenance can include only patient turns")
        if self.source_turns:
            cited_turn_ids = {
                source_id
                for fact in self.handoff.facts
                for source_id in fact.source_turn_ids
            }
            missing_sources = cited_turn_ids - set(source_ids)
            if missing_sources:
                raise ValueError(
                    "Diagnostic provenance is missing cited patient turns: "
                    f"{sorted(missing_sources)}"
                )


@dataclass(frozen=True)
class DiagnosticDraft:
    answer: str
    differential: tuple[str, ...]
    recommendations: tuple[str, ...]
    cited_fact_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _required_text(self.answer, "answer"))
        object.__setattr__(
            self, "differential", _unique_text(self.differential, "differential")
        )
        object.__setattr__(
            self,
            "recommendations",
            _unique_text(self.recommendations, "recommendations"),
        )
        object.__setattr__(
            self,
            "cited_fact_ids",
            _unique_text(self.cited_fact_ids, "cited_fact_ids"),
        )
        object.__setattr__(
            self,
            "cited_evidence_ids",
            _unique_text(self.cited_evidence_ids, "cited_evidence_ids"),
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    reason: str

    def __post_init__(self) -> None:
        if self.action not in SAFETY_ACTIONS:
            raise ValueError(f"Unsupported safety action: {self.action}")
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
