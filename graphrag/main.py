"""
GraphRAG CLI entry point.

Commands:
    build-kg          Build Neo4j knowledge graph from guidelines.txt
    query <text>      Run a single query through the ReAct agent
    benchmark         Compare Vanilla RAG vs GraphRAG on test cases
    generate-synthetic-data
                      Generate a deterministic synthetic retrieval dataset
    synthetic-benchmark
                      Compare vector-only vs hybrid retrieval on generated synthetic data
    e2e-benchmark     Evaluate final Agent answers with traces and an LLM judge
    robustness-generate
                      Design or generate the Stage 5 behavioral robustness dataset

Examples:
    python -m graphrag.main build-kg
    python -m graphrag.main query "patient has fever and cough"
    python -m graphrag.main benchmark
    python -m graphrag.main generate-synthetic-data
    python -m graphrag.main synthetic-benchmark
    python -m graphrag.main e2e-benchmark --limit 10
    python -m graphrag.main robustness-generate --dry-run
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build-kg":
        from graphrag.kg.builder import build_knowledge_graph
        build_knowledge_graph()

    elif command == "query":
        query_text = " ".join(sys.argv[2:])
        if not query_text:
            print("Usage: python -m graphrag.main query <your medical query>")
            sys.exit(1)
        from graphrag.agent.react_agent import build_agent, AgentState
        agent = build_agent()
        result = agent.invoke(AgentState(
            query=query_text,
            messages=[],
            retrieved_context=[],
            retry_count=0,
            final_answer="",
            candidate_answers=[],
            validation_verdicts=[],
            status="running",
        ))
        print("\n=== FINAL ANSWER ===")
        print(result["final_answer"])

    elif command == "benchmark":
        from graphrag.eval.benchmark import run_benchmark
        run_benchmark()

    elif command == "generate-synthetic-data":
        from graphrag.eval.synthetic_compare import save_synthetic_dataset
        path = save_synthetic_dataset()
        print(f"Synthetic dataset written to: {path}")

    elif command == "synthetic-benchmark":
        from graphrag.eval.synthetic_compare import run_retrieval_benchmark
        run_retrieval_benchmark()

    elif command == "e2e-benchmark":
        from graphrag.eval.e2e_benchmark import main as run_e2e_benchmark
        run_e2e_benchmark(sys.argv[2:])

    elif command == "robustness-generate":
        from graphrag.eval.robustness_generate import main as run_robustness_generate
        run_robustness_generate(sys.argv[2:])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
