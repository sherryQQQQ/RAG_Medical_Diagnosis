# Bounded Medical Agent

The LangGraph workflow always retrieves from Neo4j and FAISS before generation.
If the question or draft answer mentions a known Neo4j Drug entity, a safety gate
forces one contraindication lookup. Gemini then produces a grounded answer and a
reflector either approves it or requests a bounded revision.

```text
START -> graph retrieval -> vector retrieval -> optional safety check
      -> generation -> reflection -> retry or END
```

Run one question:

```bash
env LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv.nosync/bin/python -u -m graphrag.main query \
  "For mild acute pancreatitis, when should feeding begin?"
```

Run the five-case online execution smoke suite:

```bash
env LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv.nosync/bin/python -u -m graphrag.agent.smoke --limit 5 --no-resume
```

The suite is an execution and control-flow check, not a clinical-quality
benchmark. It checkpoints detailed answers, tools, retries, reflector verdicts,
and latency to `graphrag/eval/data/agent_smoke_results.json`.
