"""Injected tools used by the interactive clinical Agent.

Keeping patient interaction and retrieval behind small contracts makes the
workflow usable with a simulator, a UI, BM25, GraphRAG, or test fixtures without
changing orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from graphrag.agent.clinical_handoff import (
    ClinicalHandoff,
    ConversationTurn,
    EvidenceItem,
)


class AskPatientTool(Protocol):
    def __call__(
        self, question: str, conversation: tuple[ConversationTurn, ...]
    ) -> str: ...


class RetrieveEvidenceTool(Protocol):
    def __call__(self, handoff: ClinicalHandoff) -> tuple[EvidenceItem, ...]: ...


@dataclass(frozen=True)
class ClinicalTools:
    ask_patient: AskPatientTool
    retrieve_evidence: RetrieveEvidenceTool


def callable_patient_tool(
    responder: Callable[[str, tuple[ConversationTurn, ...]], str],
) -> AskPatientTool:
    """Name a plain callback explicitly as the patient-interaction tool."""
    return responder


def callable_retrieval_tool(
    retriever: Callable[[ClinicalHandoff], tuple[EvidenceItem, ...]],
) -> RetrieveEvidenceTool:
    """Name a plain callback explicitly as the evidence-retrieval tool."""
    return retriever
