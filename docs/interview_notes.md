# Interview Notes

Use the pattern **problem -> root cause -> engineering change -> evidence**.

## Stage 1: Make the agent runtime reproducible

- **Problem:** The repository contained an agent scaffold, but LangGraph imports
  stalled and the real FAISS path crashed.
- **Root causes:** macOS/iCloud had offloaded 42,623 virtual-environment files;
  the committed FAISS index was 384-dimensional while the configured
  `all-mpnet-base-v2` encoder returns 768 dimensions; importing FAISS before
  PyTorch on macOS/arm64 also caused a native segfault. The configured Gemini
  model was retired, and the old Neo4j Aura hostname no longer existed.
- **Changes:** Built a `.nosync` environment with pinned dependencies, reordered
  the native imports, rebuilt the 768-dimensional index, upgraded to Gemini 2.5
  Flash, created a credential-safe health check, and connected a fresh AuraDB.
- **Evidence:** LangGraph import completed in 0.35s; FAISS returned Top-5;
  Gemini and Neo4j connectivity passed; seven regression tests passed.

**Interview line:** "Before measuring agent quality, I made the runtime itself
reproducible and found that an existing index file is not evidence that the
retrieval path actually works—the encoder and index dimensions must agree."

## Stage 2: Build a trustworthy, resumable medical knowledge graph

- **Problem:** Neo4j started with zero nodes, and raw LLM extractions sometimes
  violated the intended relation schema—for example, returning
  `Disease -TREATED_BY-> Drug` instead of a Treatment endpoint.
- **Changes:** Added entity/relation allowlists, deterministic endpoint
  normalization, canonical names, transactional `MERGE` writes, uniqueness
  constraints, SHA-256 source-chunk checkpoints, read-only graph validation,
  and directed contraindication queries.
- **Failure handling:** The first strict implementation rejected whole chunks.
  It was revised to reject unknown relation types while safely normalizing the
  endpoint labels of whitelisted relations. This preserved recall without
  allowing model-generated Cypher schema values.
- **Evidence:** Processed all 15 guideline chunks. A repeat run skipped 15/15
  chunks with zero Gemini calls. The graph contains 57 Disease, 17 Symptom, 88
  Treatment, and 48 Drug nodes, plus 131 domain relationships: 76 TREATED_BY,
  35 REQUIRES_DRUG, 14 HAS_SYMPTOM, and 6 CONTRAINDICATED_WITH. Five uniqueness
  constraints and sample multi-hop paths passed validation.

**Interview line:** "I treated LLM-generated structured data as untrusted input:
the model could suggest facts, but only a deterministic schema layer could
decide what was allowed into Neo4j. I also made ingestion content-addressed, so
retries were idempotent and did not repeat paid model calls."
