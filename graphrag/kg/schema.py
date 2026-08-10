"""
Neo4j schema setup: creates constraints and indexes.

Graph model
-----------
Nodes:
    (:Disease  {name})
    (:Symptom  {name})
    (:Treatment{name})
    (:Drug     {name})

Relations:
    (:Disease)-[:HAS_SYMPTOM]->(:Symptom)
    (:Disease)-[:TREATED_BY]->(:Treatment)
    (:Treatment)-[:REQUIRES_DRUG]->(:Drug)
    (:Disease)-[:CONTRAINDICATED_WITH]->(:Drug)
"""

from neo4j import GraphDatabase
from graphrag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

CONSTRAINTS = [
    "CREATE CONSTRAINT disease_key IF NOT EXISTS FOR (d:Disease) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT symptom_key IF NOT EXISTS FOR (s:Symptom) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT treatment_key IF NOT EXISTS FOR (t:Treatment) REQUIRE t.key IS UNIQUE",
    "CREATE CONSTRAINT drug_key IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.key IS UNIQUE",
    "CREATE CONSTRAINT source_chunk_key IF NOT EXISTS FOR (s:SourceChunk) REQUIRE s.key IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX disease_idx IF NOT EXISTS FOR (d:Disease) ON (d.name)",
    "CREATE INDEX symptom_idx IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
]


def setup_schema(driver: GraphDatabase.driver) -> None:
    """Apply constraints and indexes to a fresh Neo4j database."""
    with driver.session() as session:
        for stmt in CONSTRAINTS + INDEXES:
            session.run(stmt)
    print("[schema] Constraints and indexes applied.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    setup_schema(driver)
    driver.close()
