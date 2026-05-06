"""
GraphRAG CLI entry point.

Commands:
    build-kg          Build Neo4j knowledge graph from guidelines.txt
    query <text>      Run a single query through the ReAct agent
    benchmark         Compare Vanilla RAG vs GraphRAG on test cases

Examples:
    python -m graphrag.main build-kg
    python -m graphrag.main query "patient has fever and cough"
    python -m graphrag.main benchmark
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
        ))
        print("\n=== FINAL ANSWER ===")
        print(result["final_answer"])

    elif command == "benchmark":
        from graphrag.eval.benchmark import run_benchmark
        run_benchmark()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
