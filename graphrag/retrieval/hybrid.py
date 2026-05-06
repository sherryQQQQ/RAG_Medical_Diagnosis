"""
Hybrid Retriever
----------------
Combines VectorRetriever + GraphRetriever using Reciprocal Rank Fusion (RRF).

RRF score = Σ 1 / (k + rank_i)  where k=60 (standard constant)

Text deduplication: if a vector chunk mentions a disease that graph also returned,
the context is merged rather than duplicated.

Usage:
    hr = HybridRetriever()
    context = hr.retrieve(query="fever and chest pain", symptoms=["fever", "chest pain"])
    # returns: list of HybridResult sorted by fused score
"""

from dataclasses import dataclass, field

from graphrag.retrieval.vector import VectorRetriever, VectorResult
from graphrag.retrieval.graph import GraphRetriever, GraphResult
from graphrag.config import TOP_K_VECTOR, TOP_K_GRAPH

RRF_K = 60  # standard constant for RRF


@dataclass
class HybridResult:
    text: str
    score: float
    source: str  # "vector", "graph", or "both"
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.source}, score={self.score:.3f}]\n{self.text}"


class HybridRetriever:
    def __init__(self):
        self.vector = VectorRetriever()
        self.graph = GraphRetriever()

    def retrieve(
        self,
        query: str,
        symptoms: list[str] | None = None,
        k_vector: int = TOP_K_VECTOR,
        k_graph: int = TOP_K_GRAPH,
    ) -> list[HybridResult]:
        # --- Vector retrieval ---
        vec_results: list[VectorResult] = self.vector.retrieve(query, k=k_vector)

        # --- Graph retrieval ---
        graph_results: list[GraphResult] = []
        if symptoms:
            graph_results = self.graph.retrieve(symptoms, k=k_graph)

        # --- Convert to HybridResult ---
        candidates: dict[str, HybridResult] = {}

        for rank, vr in enumerate(vec_results):
            rrf = 1.0 / (RRF_K + rank + 1)
            key = vr.text[:80]  # deduplicate by prefix
            candidates[key] = HybridResult(
                text=vr.text,
                score=rrf,
                source="vector",
                metadata={"vector_rank": rank, "vector_score": vr.score},
            )

        for rank, gr in enumerate(graph_results):
            rrf = 1.0 / (RRF_K + rank + 1)
            text = gr.to_text()
            key = gr.disease  # deduplicate by disease name
            if key in candidates:
                # Merge: add graph RRF score and upgrade source label
                candidates[key].score += rrf
                candidates[key].source = "both"
                candidates[key].text += f"\n\n[Graph context]\n{text}"
            else:
                candidates[key] = HybridResult(
                    text=text,
                    score=rrf,
                    source="graph",
                    metadata={"graph_rank": rank, "disease": gr.disease},
                )

        return sorted(candidates.values(), key=lambda r: r.score, reverse=True)

    def format_context(self, results: list[HybridResult]) -> str:
        """Render retrieved results as a single context string for the LLM."""
        sections = []
        for i, r in enumerate(results, 1):
            sections.append(f"--- Source {i} ({r.source}) ---\n{r.text}")
        return "\n\n".join(sections)
