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
env MEDICAL_RAG_LANGSMITH_TRACING=false \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv.nosync/bin/python -u -m graphrag.main query \
  "For mild acute pancreatitis, when should feeding begin?"
```

Run the five-case online execution smoke suite:

```bash
env MEDICAL_RAG_LANGSMITH_TRACING=false \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv.nosync/bin/python -u -m graphrag.agent.smoke --limit 5 --no-resume
```

The suite is an execution and control-flow check, not a clinical-quality
benchmark. It checkpoints detailed answers, tools, retries, reflector verdicts,
and latency to `graphrag/eval/data/agent_smoke_results.json`.

## Interactive Clinical Handoff Prototype

The newer research path separates interviewing from diagnosis:

```text
structured interview update -> ask patient (maximum 3 questions)
                            -> retrieval tool
                            -> fresh diagnostic packet
                            -> citation check -> safety gate
```

Each patient fact records the patient turn that supports it. The diagnostic
component receives a typed packet rather than inheriting the interviewer's
hidden context. Patient interaction, retrieval, interviewing, diagnosis, and
safety validation are injected dependencies, so fixtures and provider-backed
implementations use the same graph.

Run the zero-cost scripted demo:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m \
  graphrag.agent.clinical_demo
```

This demo validates orchestration and provenance contracts only. It does not
measure diagnostic accuracy or make a clinical-performance claim.
