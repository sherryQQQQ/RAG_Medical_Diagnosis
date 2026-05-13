"""
Medical QA Engine
-----------------
Supports two modes:
  vanilla   — baseline txtai vector RAG (existing DocumentRetriever + Gemini)
  graphrag  — Hybrid FAISS + Neo4j RRF retrieval + LangGraph ReAct agent

Usage:
    python engine.py                              # vanilla mode, rows 70-80
    python engine.py --mode graphrag              # graphrag mode
    python engine.py --mode graphrag --rows 0 10  # custom row range
"""

import argparse
import os
import sys

import pandas as pd

# Allow importing the graphrag package from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Vanilla RAG pipeline (existing txtai-based retriever + Gemini generator)
# ---------------------------------------------------------------------------

def run_vanilla(question_df: pd.DataFrame) -> list[dict]:
    from generate import Generator
    from retrieve import DocumentRetriever

    dr = DocumentRetriever()
    g = Generator()
    results = []

    for _, row in question_df.iterrows():
        question = row["prompt"]
        docs = dr.retrieve(question)
        response = g.process_query(question, docs)
        results.append({
            "question": question,
            "your_answer": response,
            "correct_answer": row["answer"],
            "documents": docs,
        })
    return results


# ---------------------------------------------------------------------------
# GraphRAG pipeline (HybridRetriever + LangGraph ReAct agent)
# ---------------------------------------------------------------------------

def _extract_symptoms(text: str) -> list[str]:
    import re
    matches = re.findall(
        r"(?:with|has|presents?|complains? of|reports?)\s+([a-z ,]+?)(?:\.|,|and|$)",
        text.lower(),
    )
    return [s.strip() for part in matches for s in part.split(",") if s.strip()][:5]


def run_graphrag(question_df: pd.DataFrame) -> list[dict]:
    from graphrag.retrieval.hybrid import HybridRetriever
    from graphrag.agent.react_agent import build_agent, AgentState

    hybrid = HybridRetriever()
    agent = build_agent()
    results = []

    for _, row in question_df.iterrows():
        question = row["prompt"]
        symptoms = _extract_symptoms(question)

        # Retrieve context via hybrid (FAISS + Neo4j RRF)
        retrieved = hybrid.retrieve(query=question, symptoms=symptoms or None)
        context_text = hybrid.format_context(retrieved)

        # Run LangGraph ReAct agent with pre-seeded context
        state = AgentState(
            query=question,
            messages=[],
            retrieved_context=[context_text],
            retry_count=0,
            final_answer="",
        )
        result = agent.invoke(state)
        answer = result.get("final_answer", "")

        results.append({
            "question": question,
            "your_answer": answer,
            "correct_answer": row["answer"],
            "documents": [(r.score, r.text) for r in retrieved],
        })
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_and_save(results: list[dict], output_file: str = "list_of_responses.txt") -> None:
    lines = []
    for r in results:
        print("Question: ", r["question"])
        print("Your Answer: ", r["your_answer"])
        print("Correct Answer: ", r["correct_answer"])
        print("Documents: ", r["documents"])
        print("--------------------------------")
        lines.append(
            "question:" + str(r["question"]) + "\n\n"
            + "your_answer:" + str(r["your_answer"]) + "\n"
            + "correct_answer:" + str(r["correct_answer"]) + "\n"
            + "documents" + str(r["documents"]) + "\n\n"
        )
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    print(f"[engine] Saved {len(results)} responses → {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical QA Engine")
    parser.add_argument(
        "--mode",
        choices=["vanilla", "graphrag"],
        default="vanilla",
        help="vanilla = txtai baseline; graphrag = FAISS+Neo4j+ReAct agent",
    )
    parser.add_argument(
        "--rows",
        nargs=2,
        type=int,
        default=[70, 80],
        metavar=("START", "END"),
        help="Row slice of medical_generalization.csv to evaluate",
    )
    args = parser.parse_args()

    question_df = pd.read_csv("medical_generalization.csv", index_col=0)
    question_df = question_df.iloc[args.rows[0]: args.rows[1], :]

    print(f"[engine] mode={args.mode}, rows={args.rows[0]}-{args.rows[1]}, n={len(question_df)}")

    if args.mode == "vanilla":
        results = run_vanilla(question_df)
    else:
        results = run_graphrag(question_df)

    print_and_save(results)
