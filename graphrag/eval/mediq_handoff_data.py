"""Pinned MediQ pilot data and a zero-call patient-fact tool."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from graphrag.agent.clinical_handoff import ConversationTurn


DEFAULT_SPEC = Path(__file__).parent / "specs" / "mediq_handoff_pilot.json"
DEFAULT_ROOT = Path(__file__).parent / "external" / "mediq"
DEFAULT_SOURCE = DEFAULT_ROOT / "all_dev_good.jsonl"
DIAGNOSTIC_CONDITIONS = (
    "raw-conversation",
    "structured-handoff",
    "handoff-plus-sources",
)


@dataclass(frozen=True)
class MediQCase:
    case_id: str
    source_id: int
    specialty: str
    question: str
    initial_info: str
    context: tuple[str, ...]
    facts: tuple[str, ...]
    options: dict[str, str]
    answer_choice: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "revision",
        "source_path",
        "source_sha256",
        "license",
        "seed",
        "specialties",
        "selected_source_ids",
        "max_questions",
        "retrieval",
        "diagnostic_conditions",
    }
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError("Invalid MediQ handoff pilot spec")
    if spec["format_version"] != 1 or spec["license"] != "CC-BY-4.0":
        raise ValueError("Unsupported MediQ handoff pilot spec")
    if spec["diagnostic_conditions"] != list(DIAGNOSTIC_CONDITIONS):
        raise ValueError("Unexpected diagnostic conditions")
    if spec["max_questions"] != 3 or spec["retrieval"].get("top_k") != 8:
        raise ValueError("Pilot question or retrieval budget changed")
    return spec


def source_url(spec: Mapping[str, Any]) -> str:
    return (
        "https://raw.githubusercontent.com/stellalisy/mediQ/"
        f"{spec['revision']}/{spec['source_path']}"
    )


def download_source(
    destination: Path = DEFAULT_SOURCE, spec: Mapping[str, Any] | None = None
) -> Path:
    spec = dict(spec or load_spec())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != spec["source_sha256"]:
            raise ValueError("Existing MediQ source failed the pinned SHA-256 check")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(source_url(spec), headers={"User-Agent": "medical-agent-eval/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha256(temporary) != spec["source_sha256"]:
            raise ValueError("Downloaded MediQ source failed the pinned SHA-256 check")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _strip_fact_index(text: str) -> str:
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", str(text)).strip()


def load_cases(
    source_path: Path = DEFAULT_SOURCE, spec: Mapping[str, Any] | None = None
) -> list[MediQCase]:
    spec = dict(spec or load_spec())
    if sha256(source_path) != spec["source_sha256"]:
        raise ValueError("MediQ source SHA-256 does not match the pinned spec")
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
    ]
    rng = random.Random(int(spec["seed"]))
    selected: list[Mapping[str, Any]] = []
    for specialty in spec["specialties"]:
        candidates = [
            row
            for row in rows
            if row.get("patient", {}).get("gpt_specialty") == specialty
        ]
        if not candidates:
            raise ValueError(f"MediQ specialty is empty: {specialty}")
        selected.append(rng.choice(candidates))
    ids = [int(row["id"]) for row in selected]
    if ids != spec["selected_source_ids"]:
        raise ValueError(f"MediQ fixed selection changed: {ids}")

    cases: list[MediQCase] = []
    for specialty, row in zip(spec["specialties"], selected):
        context = tuple(
            str(value).strip() for value in row["context"] if str(value).strip()
        )
        facts = tuple(
            _strip_fact_index(value)
            for value in row.get("facts", [])
            if _strip_fact_index(value)
        )
        options = {str(key): str(value) for key, value in row["options"].items()}
        answer_choice = str(row["answer_idx"]).upper()
        if not context or not facts or answer_choice not in options:
            raise ValueError(f"Invalid selected MediQ case: {row.get('id')}")
        cases.append(
            MediQCase(
                case_id=f"mediq-{row['id']}",
                source_id=int(row["id"]),
                specialty=str(specialty),
                question=str(row["question"]).strip(),
                initial_info=context[0],
                context=context,
                facts=facts,
                options=options,
                answer_choice=answer_choice,
            )
        )
    return cases


def dataset_fingerprint(cases: list[MediQCase], spec: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "source_sha256": spec["source_sha256"],
            "seed": spec["seed"],
            "ids": [case.source_id for case in cases],
            "conditions": list(DIAGNOSTIC_CONDITIONS),
        }
    )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "can", "did", "do",
    "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "of",
    "on", "or", "patient", "please", "that", "the", "their", "they", "this", "to",
    "was", "were", "what", "when", "where", "which", "with", "you", "your",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def token_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


CLINICAL_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "allergy": ("allergy", "allergies", "allergic", "reaction"),
    "family_history": ("family history", "mother", "father", "sibling", "familial"),
    "medical_history": (
        "medical history", "past history", "medical condition", "medical conditions",
        "ongoing health problem", "ongoing health problems", "comorbidity", "diabetes",
        "hypertension", "asthma", "cancer", "immunocompromised",
    ),
    "medication": (
        "medication", "medications", "medicine", "drug", "drugs",
        "prescription", "dose", "treatment", "antibiotic",
    ),
    "reproductive": (
        "pregnant", "pregnancy", "menstrual", "period", "lmp",
        "contraception", "iud", "sexual activity", "sexually active",
        "sexual intercourse", "unprotected sex", "vaginal", "douche", "douching",
    ),
    "social_history": (
        "social history", "smoking", "smoke", "alcohol", "recreational drug",
        "occupation", "travel", "exposure", "living situation",
    ),
    "symptom": (
        "symptom", "symptoms", "pain", "fever", "cough", "nausea", "vomiting",
        "diarrhea", "dizziness", "weakness", "fatigue", "rash", "swelling",
        "bleeding", "discharge", "shortness of breath", "headache", "seizure",
    ),
    "timeline": (
        "when", "how long", "onset", "duration", "started", "ago", "day",
        "days", "week", "weeks", "month", "months", "year", "years",
    ),
    "trauma": (
        "injury", "injuries", "trauma", "accident", "fall", "wound",
        "laceration", "fracture", "burn", "hit", "struck",
    ),
    "vital_sign": (
        "vital", "vitals", "vital signs", "temperature", "blood pressure",
        "heart rate", "pulse", "respiratory rate", "oxygen saturation", "spo2",
    ),
}


def clinical_concepts(text: str) -> set[str]:
    """Map free text to a small auditable clinical-intent vocabulary."""
    lowered = " ".join(str(text).lower().split())

    def contains_phrase(phrase: str) -> bool:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+")
        return re.search(pattern + r"(?![a-z0-9])", lowered) is not None

    concepts = {
        concept
        for concept, phrases in CLINICAL_CONCEPT_LEXICON.items()
        if any(contains_phrase(phrase) for phrase in phrases)
    }
    if re.search(r"\b(?:bp|hr|rr)\s*[:=]?\s*\d", lowered):
        concepts.add("vital_sign")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:°?[fc]|mmhg|bpm|%)\b", lowered):
        concepts.add("vital_sign")
    return concepts


class DeterministicFactPatient:
    """Zero-call approximation of MediQ's fact-select patient."""

    matcher_version = "lexical-v1"

    def __init__(self, case: MediQCase):
        self.case = case
        self.revealed: dict[str, str] = {}
        self.revealed_by_turn: dict[str, list[str]] = {"patient-0": []}
        for index, fact in enumerate(case.facts, start=1):
            fact_id = f"source-fact-{index}"
            if token_f1(fact, case.initial_info) >= 0.45:
                self.revealed[fact_id] = fact
                self.revealed_by_turn["patient-0"].append(fact_id)

    def _score(self, question: str, fact: str) -> float:
        question_tokens = _tokens(question)
        intent_tokens: set[str] = set()
        lowered = question.lower()
        if "when" in lowered or "how long" in lowered:
            intent_tokens |= {
                "time", "day", "days", "week", "weeks", "month", "months",
                "year", "years", "ago",
            }
        if "medication" in lowered or "drug" in lowered:
            intent_tokens |= {"medication", "medications", "drug", "drugs", "treatment"}
        if "history" in lowered:
            intent_tokens |= {"history", "previous", "past"}
        query = question_tokens | intent_tokens
        fact_tokens = _tokens(fact)
        return len(query & fact_tokens) / math.sqrt(max(1, len(fact_tokens)))

    def __call__(
        self, question: str, conversation: tuple[ConversationTurn, ...]
    ) -> str:

        ranked: list[tuple[float, str, str]] = []
        for index, fact in enumerate(self.case.facts, start=1):
            fact_id = f"source-fact-{index}"
            if fact_id in self.revealed:
                continue
            score = self._score(question, fact)
            ranked.append((score, fact_id, fact))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [item for item in ranked[:2] if item[0] > 0]
        if not selected:
            return "The patient cannot answer this question from the available record."
        next_patient_turn = (
            f"patient-{sum(turn.role == 'agent' for turn in conversation) + 1}"
        )
        for _, fact_id, fact in selected:
            self.revealed[fact_id] = fact
            self.revealed_by_turn.setdefault(next_patient_turn, []).append(fact_id)
        return " ".join(fact for _, _, fact in selected)


class ConceptAwareFactPatient(DeterministicFactPatient):
    """Zero-call patient tool combining lexical overlap with clinical intent."""

    matcher_version = "clinical-concept-v2"

    def _score(self, question: str, fact: str) -> float:
        lexical_score = super()._score(question, fact)
        shared_concepts = clinical_concepts(question) & clinical_concepts(fact)
        return lexical_score + 0.85 * len(shared_concepts)


class CoverageAwareQuestionRefiner:
    """Diversify repeated questions using an auditable clinical checklist."""

    version = "clinical-coverage-v1"
    _templates = (
        ("timeline", "When did this start, and how has it changed over time?"),
        ("symptom", "What other symptoms have you noticed?"),
        (
            "medical_history",
            "Do you have any important past medical conditions or ongoing health problems?",
        ),
        ("medication", "What medications or recent treatments are you taking?"),
        ("allergy", "Do you have any medication or other allergies?"),
        ("vital_sign", "What are the most recent vital signs, if known?"),
        ("trauma", "Was there any recent injury, fall, or other trauma?"),
        ("social_history", "Are there relevant smoking, alcohol, travel, or exposure risks?"),
        ("family_history", "Is there any relevant family history?"),
    )

    def __call__(
        self,
        question: str,
        conversation: tuple[ConversationTurn, ...],
        _handoff: Any,
    ) -> str:
        prior_agent_turns = [turn for turn in conversation if turn.role == "agent"]
        if not prior_agent_turns:
            return question
        asked = set().union(
            *(clinical_concepts(turn.content) for turn in prior_agent_turns)
        )
        proposed = clinical_concepts(question)
        if not proposed or proposed - asked:
            return question
        covered = asked | set().union(
            *(clinical_concepts(turn.content) for turn in conversation)
        )
        for concept, template in self._templates:
            if concept not in covered:
                return template
        return question


def clinical_tool_fingerprint() -> str:
    """Content fingerprint for freezing the holdout's deterministic tools."""
    return canonical_hash(
        {
            "matcher": ConceptAwareFactPatient.matcher_version,
            "concept_weight": 0.85,
            "lexicon": CLINICAL_CONCEPT_LEXICON,
            "refiner": CoverageAwareQuestionRefiner.version,
            "question_templates": CoverageAwareQuestionRefiner._templates,
        }
    )
