"""Fast, credential-safe health check for the Agentic GraphRAG runtime."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def timed(name, check):
    started = time.perf_counter()
    try:
        detail = check()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"FAIL {name:<18} {elapsed:>7.2f}s  {type(exc).__name__}: {exc}")
        return False
    elapsed = time.perf_counter() - started
    print(f"PASS {name:<18} {elapsed:>7.2f}s  {detail}")
    return True


def check_imports():
    from langchain_core.messages import ToolMessage  # noqa: F401
    from langgraph.graph import StateGraph  # noqa: F401

    return "LangChain and LangGraph import successfully"


def check_faiss_files():
    import faiss

    prefix = ROOT / "graphrag" / "db" / "faiss_index"
    required = [prefix.with_suffix(".faiss"), prefix.with_suffix(".pkl")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    index = faiss.read_index(str(required[0]))
    if index.d != 768:
        raise RuntimeError(f"FAISS dimension {index.d} does not match all-mpnet-base-v2 (768)")
    if index.ntotal <= 0:
        raise RuntimeError("FAISS index contains no vectors")
    return f"dimension={index.d}, vectors={index.ntotal}"


def check_gemini():
    from google import genai

    api_key = os.environ["GOOGLE_API_KEY"]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents="Reply with exactly: OK")
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return f"model={model}, non-empty response"


def check_neo4j():
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            record = session.run(
                "MATCH (n) RETURN count(n) AS nodes"
            ).single(strict=True)
        return f"connected, nodes={int(record['nodes'])}"
    finally:
        driver.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online", action="store_true", help="also make one Gemini and one Neo4j request"
    )
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")

    checks = [
        ("python", lambda: sys.version.split()[0]),
        ("imports", check_imports),
        ("faiss files", check_faiss_files),
    ]
    if args.online:
        checks.extend((("gemini", check_gemini), ("neo4j", check_neo4j)))

    outcomes = [timed(name, check) for name, check in checks]
    passed = all(outcomes)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
