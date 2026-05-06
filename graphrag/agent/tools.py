"""
LangChain tool definitions for the ReAct agent.

Three tools:
  1. retrieve_vector   — semantic FAISS search over guidelines
  2. retrieve_graph    — Neo4j multi-hop symptom→disease→treatment lookup
  3. check_contraindications — Neo4j drug safety check
"""

import json
from typing import Annotated

from langchain_core.tools import tool

from graphrag.retrieval.vector import VectorRetriever
from graphrag.retrieval.graph import GraphRetriever

# Singletons — initialised once, reused across tool calls
_vector_retriever: VectorRetriever | None = None
_graph_retriever: GraphRetriever | None = None


def _get_vector() -> VectorRetriever:
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorRetriever()
    return _vector_retriever


def _get_graph() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


@tool
def retrieve_vector(query: Annotated[str, "Free-text medical query to search guidelines"]) -> str:
    """
    Semantic search over the medical guidelines using FAISS + Sentence-BERT.
    Use this for open-ended questions or when you don't have specific symptom names.
    Returns the top matching text passages.
    """
    results = _get_vector().retrieve(query, k=5)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(
        f"[Passage {i+1}, score={r.score:.3f}]\n{r.text}"
        for i, r in enumerate(results)
    )


@tool
def retrieve_graph(
    symptoms: Annotated[
        str,
        "JSON list of symptom strings, e.g. '[\"fever\", \"cough\", \"chest pain\"]'"
    ]
) -> str:
    """
    Multi-hop knowledge graph lookup via Neo4j.
    Given a list of symptoms, traverses Disease→Treatment→Drug paths.
    Use this when you have identified specific symptoms to look up structured diagnostic paths.
    Returns diseases, treatments, drugs, and contraindications.
    """
    try:
        symptom_list: list[str] = json.loads(symptoms)
    except json.JSONDecodeError:
        symptom_list = [s.strip() for s in symptoms.split(",")]

    results = _get_graph().retrieve(symptom_list, k=3)
    if not results:
        return "No matching diseases found in knowledge graph for the given symptoms."
    return "\n\n".join(r.to_text() for r in results)


@tool
def check_contraindications(
    drug: Annotated[str, "Name of the drug to check"],
    conditions: Annotated[
        str,
        "JSON list of patient conditions/diseases, e.g. '[\"diabetes\", \"hypertension\"]'"
    ],
) -> str:
    """
    Safety check: determines whether a drug is contraindicated given a patient's conditions.
    Use this before recommending a specific medication.
    Returns contraindication warnings or confirms the drug is safe for the given conditions.
    """
    try:
        condition_list: list[str] = json.loads(conditions)
    except json.JSONDecodeError:
        condition_list = [c.strip() for c in conditions.split(",")]

    warnings = _get_graph().check_contraindications(drug, condition_list)
    if not warnings:
        return f"No contraindications found for '{drug}' with the given conditions."
    return "⚠️ Contraindication warnings:\n" + "\n".join(f"  - {w}" for w in warnings)


# Exported tool list for the agent
TOOLS = [retrieve_vector, retrieve_graph, check_contraindications]
