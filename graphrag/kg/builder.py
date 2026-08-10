"""
Knowledge Graph Builder
-----------------------
Pipeline:
  1. Read guidelines.txt → split into paragraph chunks
  2. For each chunk, call Gemini to extract structured entities + relations (JSON)
  3. MERGE nodes/relations into Neo4j

Run once to populate the graph:
    python -m graphrag.kg.builder
"""

import json
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import google.genai as genai
from google.genai import types
from neo4j import GraphDatabase

from graphrag.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GUIDELINES_PATH,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)
from graphrag.kg.schema import setup_schema

# ---------- helpers ----------------------------------------------------------

EXTRACT_PROMPT = """You are a medical knowledge graph extractor.

From the following medical text, extract all medical entities and their relationships.

Rules:
- Entity types: Disease, Symptom, Treatment, Drug
- Relation types: HAS_SYMPTOM, TREATED_BY, REQUIRES_DRUG, CONTRAINDICATED_WITH
- Valid relation directions:
  Disease -HAS_SYMPTOM-> Symptom
  Disease -TREATED_BY-> Treatment
  Treatment -REQUIRES_DRUG-> Drug
  Disease -CONTRAINDICATED_WITH-> Drug
- Use short canonical names (e.g. "fever" not "high body temperature")
- Only extract what is explicitly stated

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "Disease|Symptom|Treatment|Drug"}}
  ],
  "relations": [
    {{"from": "...", "from_type": "...", "relation": "HAS_SYMPTOM|TREATED_BY|REQUIRES_DRUG|CONTRAINDICATED_WITH", "to": "...", "to_type": "..."}}
  ]
}}

Text:
{text}
"""

ENTITY_TYPES = {"Disease", "Symptom", "Treatment", "Drug"}
RELATION_SCHEMA = {
    "HAS_SYMPTOM": ("Disease", "Symptom"),
    "TREATED_BY": ("Disease", "Treatment"),
    "REQUIRES_DRUG": ("Treatment", "Drug"),
    "CONTRAINDICATED_WITH": ("Disease", "Drug"),
}

MERGE_NODE_QUERY = """
MERGE (n:{label} {{key: $key}})
ON CREATE SET n.name = $name
ON MATCH SET n.name = $name
"""
MERGE_REL_QUERY = """
MERGE (a:{from_label} {{key: $from_key}})
ON CREATE SET a.name = $from_name
MERGE (b:{to_label} {{key: $to_key}})
ON CREATE SET b.name = $to_name
MERGE (a)-[:{rel_type}]->(b)
"""


@dataclass(frozen=True)
class BuildReport:
    chunks: int
    succeeded: int
    failed: int
    skipped: int


class InvalidExtraction(ValueError):
    """Raised when model output does not match the whitelisted graph schema."""


def chunk_guidelines(path: str, min_chars: int = 100) -> list[str]:
    """Split guidelines into paragraph-level chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]


def _canonical_name(value: Any) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not name or len(name) > 160:
        raise InvalidExtraction(f"Invalid entity name: {value!r}")
    return name


def normalize_extraction(extraction: dict) -> dict:
    """Validate model output and return a canonical, deduplicated extraction."""
    if not isinstance(extraction, dict):
        raise InvalidExtraction("Extraction must be a JSON object")

    entities: dict[tuple[str, str], dict[str, str]] = {}
    for entity in extraction.get("entities", []):
        if not isinstance(entity, dict) or entity.get("type") not in ENTITY_TYPES:
            raise InvalidExtraction(f"Unknown entity type: {entity!r}")
        name = _canonical_name(entity.get("name"))
        label = entity["type"]
        entities[(label, name)] = {"name": name, "type": label}

    relations: dict[tuple[str, str, str], dict[str, str]] = {}
    corrections: list[str] = []
    for relation in extraction.get("relations", []):
        if not isinstance(relation, dict):
            raise InvalidExtraction(f"Invalid relation: {relation!r}")
        rel_type = relation.get("relation")
        expected = RELATION_SCHEMA.get(rel_type)
        if expected is None:
            raise InvalidExtraction(f"Unknown relation type: {rel_type!r}")

        from_type, to_type = relation.get("from_type"), relation.get("to_type")
        from_name = _canonical_name(relation.get("from"))
        to_name = _canonical_name(relation.get("to"))
        if (from_type, to_type) == tuple(reversed(expected)):
            from_type, to_type = to_type, from_type
            from_name, to_name = to_name, from_name
            corrections.append(f"reversed {rel_type} endpoints")
        elif (from_type, to_type) != expected:
            original_types = f"{from_type}->{to_type}"
            # Relation names are whitelisted, so endpoint labels can be safely
            # coerced to the declared schema without interpolating model output.
            from_type, to_type = expected
            corrections.append(
                f"coerced {rel_type} endpoints from {original_types} to "
                f"{from_type}->{to_type}"
            )

        entities[(from_type, from_name)] = {"name": from_name, "type": from_type}
        entities[(to_type, to_name)] = {"name": to_name, "type": to_type}
        key = (from_name, rel_type, to_name)
        relations[key] = {
            "from": from_name,
            "from_type": from_type,
            "relation": rel_type,
            "to": to_name,
            "to_type": to_type,
        }

    return {
        "entities": list(entities.values()),
        "relations": list(relations.values()),
        "corrections": corrections,
    }


def extract_entities_relations(chunk: str, client: Any, max_retries: int = 4) -> dict:
    """Call Gemini to extract entities/relations from a text chunk, with retry on 429."""
    prompt = EXTRACT_PROMPT.format(text=chunk)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            extraction = normalize_extraction(json.loads(raw))
            for correction in extraction["corrections"]:
                print(f"  [schema-correction] {correction}", file=sys.stderr)
            return extraction
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                print(f"  [rate-limit] 429 hit, waiting {wait}s before retry {attempt + 1}/{max_retries - 1}...", file=sys.stderr)
                time.sleep(wait)
            elif isinstance(e, (json.JSONDecodeError, InvalidExtraction)) and attempt < max_retries - 1:
                print(
                    f"  [invalid-output] retry {attempt + 1}/{max_retries - 1}: {e}",
                    file=sys.stderr,
                )
            else:
                raise


def load_to_neo4j(driver, extraction: dict, source_key: str | None = None) -> None:
    """Write extracted entities and relations into Neo4j."""
    extraction = normalize_extraction(extraction)

    def write_extraction(tx) -> None:
        for entity in extraction["entities"]:
            tx.run(
                MERGE_NODE_QUERY.format(label=entity["type"]),
                key=entity["name"],
                name=entity["name"],
            )

        for rel in extraction["relations"]:
            query = MERGE_REL_QUERY.format(
                from_label=rel["from_type"],
                to_label=rel["to_type"],
                rel_type=rel["relation"],
            )
            tx.run(
                query,
                from_key=rel["from"],
                from_name=rel["from"],
                to_key=rel["to"],
                to_name=rel["to"],
            )
        if source_key:
            tx.run("MERGE (:SourceChunk {key: $key})", key=source_key)

    with driver.session() as session:
        session.execute_write(write_extraction)


_INTER_CHUNK_DELAY = 3  # seconds between chunks to stay within free-tier RPM


def _source_key(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()


def _is_processed(driver, source_key: str) -> bool:
    with driver.session() as session:
        record = session.run(
            "MATCH (s:SourceChunk {key: $key}) RETURN count(s) AS count",
            key=source_key,
        ).single(strict=True)
    return bool(record["count"])


def build_knowledge_graph() -> BuildReport:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    print("[builder] Setting up schema...")
    setup_schema(driver)

    chunks = chunk_guidelines(GUIDELINES_PATH)
    print(f"[builder] Processing {len(chunks)} chunks from guidelines.txt...")

    success, failed, skipped = 0, 0, 0
    for i, chunk in enumerate(chunks):
        source_key = _source_key(chunk)
        if _is_processed(driver, source_key):
            skipped += 1
            print(f"  [{i+1}/{len(chunks)}] already processed; skipping")
            continue
        try:
            extraction = extract_entities_relations(chunk, client)
            load_to_neo4j(driver, extraction, source_key=source_key)
            success += 1
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(chunks)}] chunks processed")
        except Exception as e:
            failed += 1
            print(f"  [WARN] chunk {i} failed: {e}", file=sys.stderr)
        if i < len(chunks) - 1:
            time.sleep(_INTER_CHUNK_DELAY)

    driver.close()
    print(f"[builder] Done — {success} succeeded, {failed} failed, {skipped} skipped.")
    return BuildReport(
        chunks=len(chunks), succeeded=success, failed=failed, skipped=skipped
    )


if __name__ == "__main__":
    build_knowledge_graph()
