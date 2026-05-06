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
import re
import sys
from typing import Any

import google.generativeai as genai
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

MERGE_NODE_QUERY = "MERGE (n:{label} {{name: $name}})"
MERGE_REL_QUERY = """
MERGE (a:{from_label} {{name: $from_name}})
MERGE (b:{to_label} {{name: $to_name}})
MERGE (a)-[:{rel_type}]->(b)
"""


def chunk_guidelines(path: str, min_chars: int = 100) -> list[str]:
    """Split guidelines into paragraph-level chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]


def extract_entities_relations(chunk: str, model: Any) -> dict:
    """Call Gemini to extract entities/relations from a text chunk."""
    prompt = EXTRACT_PROMPT.format(text=chunk)
    response = model.generate_content(prompt)
    raw = response.text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def load_to_neo4j(driver, extraction: dict) -> None:
    """Write extracted entities and relations into Neo4j."""
    with driver.session() as session:
        # Nodes
        for entity in extraction.get("entities", []):
            label = entity["type"]
            session.run(MERGE_NODE_QUERY.format(label=label), name=entity["name"])

        # Relations
        for rel in extraction.get("relations", []):
            query = MERGE_REL_QUERY.format(
                from_label=rel["from_type"],
                to_label=rel["to_type"],
                rel_type=rel["relation"],
            )
            session.run(query, from_name=rel["from"], to_name=rel["to"])


def build_knowledge_graph() -> None:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    print("[builder] Setting up schema...")
    setup_schema(driver)

    chunks = chunk_guidelines(GUIDELINES_PATH)
    print(f"[builder] Processing {len(chunks)} chunks from guidelines.txt...")

    success, failed = 0, 0
    for i, chunk in enumerate(chunks):
        try:
            extraction = extract_entities_relations(chunk, model)
            load_to_neo4j(driver, extraction)
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(chunks)}] chunks processed")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            failed += 1
            print(f"  [WARN] chunk {i} failed: {e}", file=sys.stderr)

    driver.close()
    print(f"[builder] Done — {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    build_knowledge_graph()
