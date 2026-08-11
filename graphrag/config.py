"""Central config: loads .env and exposes typed constants."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Google Gemini ---
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_REQUEST_TIMEOUT_S: float = float(
    os.getenv("GEMINI_REQUEST_TIMEOUT_S", "30")
)
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "1"))

# --- Neo4j ---
NEO4J_URI: str = os.environ["NEO4J_URI"]
NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD: str = os.environ["NEO4J_PASSWORD"]

# --- LangSmith (optional but recommended) ---
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "medical-graphrag")
LANGSMITH_TRACING: bool = (
    os.getenv("MEDICAL_RAG_LANGSMITH_TRACING", "false").lower() == "true"
)
if LANGSMITH_API_KEY and LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# --- Retrieval ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K_VECTOR: int = int(os.getenv("TOP_K_VECTOR", "5"))
TOP_K_GRAPH: int = int(os.getenv("TOP_K_GRAPH", "3"))

# --- Paths ---
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
GUIDELINES_PATH = os.path.join(REPO_ROOT, "final", "guidelines.txt")
EVAL_CSV_PATH = os.path.join(REPO_ROOT, "final", "medical_generalization.csv")
FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "graphrag", "db", "faiss_index")
