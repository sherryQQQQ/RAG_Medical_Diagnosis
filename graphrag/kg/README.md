# Medical Knowledge Graph

The builder sends paragraph chunks from `final/guidelines.txt` to the configured
Gemini model, validates the returned JSON against a fixed medical graph schema,
and transactionally merges the normalized entities and relations into Neo4j.

```bash
env MEDICAL_RAG_LANGSMITH_TRACING=false \
  .venv.nosync/bin/python -u -m graphrag.kg.builder

.venv.nosync/bin/python -m graphrag.kg.validate

.venv.nosync/bin/python -m unittest \
  graphrag.kg.test_builder graphrag.retrieval.test_graph -v
```

Successful chunks receive a SHA-256 `SourceChunk` marker. Re-running the builder
skips those chunks, making interruption recovery idempotent and avoiding repeat
Gemini calls. The validation command is read-only and checks label coverage,
core relation coverage, unique constraints, and sample multi-hop paths.

The current measured graph statistics are saved in
`graphrag/eval/data/kg_validation_results.json`.
