"""
Neo4j graph retriever
---------------------
Multi-hop Cypher traversal:
  Symptom → Disease → Treatment / Drug / Contraindication

This is the key differentiator vs vanilla RAG — structured relational
knowledge lets us answer "what disease causes X and what are the
contraindications of its treatment?" in a single graph query.

Usage:
    gr = GraphRetriever()
    results = gr.retrieve(["fever", "cough"])
    # returns: list of GraphResult (structured + text form)
"""

from dataclasses import dataclass, field

from neo4j import GraphDatabase

from graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, TOP_K_GRAPH

# Two-hop: symptom → disease → treatment (+ optional drug)
SYMPTOM_TO_DISEASE_QUERY = """
MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
WHERE toLower(s.name) IN $symptoms
WITH d, collect(s.name) AS matched_symptoms
MATCH (d)-[:TREATED_BY]->(t:Treatment)
OPTIONAL MATCH (t)-[:REQUIRES_DRUG]->(dr:Drug)
OPTIONAL MATCH (d)-[:CONTRAINDICATED_WITH]->(ci:Drug)
RETURN d.name AS disease,
       matched_symptoms,
       collect(DISTINCT t.name) AS treatments,
       collect(DISTINCT dr.name) AS drugs,
       collect(DISTINCT ci.name) AS contraindications
ORDER BY size(matched_symptoms) DESC
LIMIT $limit
"""

# Fallback: if no symptom match, search disease name directly. This query avoids
# HAS_SYMPTOM so it still works when the KG was built without symptom edges.
DISEASE_SEARCH_QUERY = """
MATCH (d:Disease)
WHERE toLower(d.name) CONTAINS $keyword
OPTIONAL MATCH (d)-[:TREATED_BY]->(t)
OPTIONAL MATCH (t)-[:REQUIRES_DRUG]->(dr:Drug)
OPTIONAL MATCH (d)-[:CONTRAINDICATED_WITH]->(ci)
RETURN d.name AS disease,
       [] AS symptoms,
       collect(DISTINCT t.name) AS treatments,
       collect(DISTINCT dr.name) AS drugs,
       collect(DISTINCT ci.name) AS contraindications
LIMIT $limit
"""

ENTITY_SEARCH_QUERY = """
MATCH (d:Disease)
WHERE any(term IN $terms WHERE toLower(d.name) CONTAINS term OR term CONTAINS toLower(d.name))
OPTIONAL MATCH (d)-[:TREATED_BY]->(t)
OPTIONAL MATCH (t)-[:REQUIRES_DRUG]->(dr:Drug)
OPTIONAL MATCH (d)-[:CONTRAINDICATED_WITH]->(ci)
RETURN d.name AS disease,
       [] AS matched_symptoms,
       collect(DISTINCT t.name) AS treatments,
       collect(DISTINCT dr.name) AS drugs,
       collect(DISTINCT ci.name) AS contraindications
LIMIT $limit
"""

CONTRAINDICATION_QUERY = """
MATCH (d:Disease)-[:CONTRAINDICATED_WITH]->(dr:Drug)
WHERE toLower(dr.name) CONTAINS $drug
  AND any(condition IN $conditions WHERE toLower(d.name) CONTAINS condition)
RETURN DISTINCT d.name AS disease, dr.name AS drug
"""


@dataclass
class GraphResult:
    disease: str
    matched_symptoms: list[str] = field(default_factory=list)
    treatments: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"Disease: {self.disease}"]
        if self.matched_symptoms:
            lines.append(f"  Symptoms: {', '.join(self.matched_symptoms)}")
        if self.treatments:
            lines.append(f"  Treatments: {', '.join(self.treatments)}")
        if self.drugs:
            lines.append(f"  Drugs: {', '.join(self.drugs)}")
        if self.contraindications:
            lines.append(f"  Contraindications: {', '.join(self.contraindications)}")
        return "\n".join(lines)


class GraphRetriever:
    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self._relationship_types: set[str] | None = None

    def retrieve(self, symptoms: list[str], k: int = TOP_K_GRAPH) -> list[GraphResult]:
        """Primary retrieval: match by symptoms."""
        normalized = [s.lower().strip() for s in symptoms if s.strip()]
        if not normalized:
            return []

        results = self._entity_search(normalized, k)
        if results:
            return results

        if not self._relationship_exists("HAS_SYMPTOM"):
            return self._fallback_search(normalized[0], k)

        with self._driver.session() as session:
            records = session.run(
                SYMPTOM_TO_DISEASE_QUERY,
                symptoms=normalized,
                limit=k,
            )
            results = [
                GraphResult(
                    disease=r["disease"],
                    matched_symptoms=r["matched_symptoms"],
                    treatments=r["treatments"],
                    drugs=r["drugs"],
                    contraindications=r["contraindications"],
                )
                for r in records
            ]
        if not results:
            results = self._fallback_search(normalized[0] if normalized else "", k)
        return results

    def check_contraindications(self, drug: str, conditions: list[str]) -> list[str]:
        """Check if a drug is contraindicated for any of the given conditions."""
        normalized_conditions = [c.lower().strip() for c in conditions if c.strip()]
        if not normalized_conditions:
            return []
        with self._driver.session() as session:
            records = session.run(
                CONTRAINDICATION_QUERY,
                drug=drug.lower().strip(),
                conditions=normalized_conditions,
            )
            return [f"{r['drug']} contraindicated for {r['disease']}" for r in records]

    def _relationship_exists(self, rel_type: str) -> bool:
        if self._relationship_types is None:
            with self._driver.session() as session:
                records = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                self._relationship_types = {r["relationshipType"] for r in records}
        return rel_type in self._relationship_types

    def _entity_search(self, terms: list[str], k: int) -> list[GraphResult]:
        with self._driver.session() as session:
            records = session.run(ENTITY_SEARCH_QUERY, terms=terms, limit=k)
            return [
                GraphResult(
                    disease=r["disease"],
                    matched_symptoms=r["matched_symptoms"],
                    treatments=r["treatments"],
                    drugs=r["drugs"],
                    contraindications=r["contraindications"],
                )
                for r in records
            ]

    def _fallback_search(self, keyword: str, k: int) -> list[GraphResult]:
        with self._driver.session() as session:
            records = session.run(DISEASE_SEARCH_QUERY, keyword=keyword, limit=k)
            return [
                GraphResult(
                    disease=r["disease"],
                    matched_symptoms=r["symptoms"],
                    treatments=r["treatments"],
                    drugs=r["drugs"],
                    contraindications=r["contraindications"],
                )
                for r in records
            ]

    def close(self):
        self._driver.close()
