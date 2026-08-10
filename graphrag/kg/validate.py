"""Read-only validation report for the populated Neo4j medical graph."""

from __future__ import annotations

import json

from neo4j import GraphDatabase

from graphrag.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


COUNT_NODES = """
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY label
"""

COUNT_RELATIONSHIPS = """
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY type
"""

SAMPLE_PATHS = """
MATCH (d:Disease)-[:TREATED_BY]->(t:Treatment)
OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (t)-[:REQUIRES_DRUG]->(dr:Drug)
RETURN d.name AS disease,
       collect(DISTINCT s.name)[..3] AS symptoms,
       t.name AS treatment,
       collect(DISTINCT dr.name)[..3] AS drugs
LIMIT 5
"""


def collect_graph_report(driver) -> dict:
    with driver.session() as session:
        node_counts = {r["label"]: r["count"] for r in session.run(COUNT_NODES)}
        relationship_counts = {r["type"]: r["count"] for r in session.run(COUNT_RELATIONSHIPS)}
        sample_paths = [dict(record) for record in session.run(SAMPLE_PATHS)]
        constraints = [record["name"] for record in session.run("SHOW CONSTRAINTS YIELD name")]

    required_labels = {"Disease", "Symptom", "Treatment", "Drug"}
    required_relations = {"HAS_SYMPTOM", "TREATED_BY", "REQUIRES_DRUG"}
    checks = {
        "has_all_labels": required_labels.issubset(node_counts),
        "has_core_relations": required_relations.issubset(relationship_counts),
        "has_sample_paths": bool(sample_paths),
        "has_unique_constraints": all(f"{label.lower()}_key" in constraints for label in required_labels),
    }
    return {
        "node_counts": node_counts,
        "relationship_counts": relationship_counts,
        "constraints": constraints,
        "sample_paths": sample_paths,
        "checks": checks,
        "valid": all(checks.values()),
    }


def main() -> None:
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    try:
        report = collect_graph_report(driver)
    finally:
        driver.close()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
