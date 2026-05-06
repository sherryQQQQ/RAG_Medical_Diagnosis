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
    "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT treatment_name IF NOT EXISTS FOR (t:Treatment) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE",
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
