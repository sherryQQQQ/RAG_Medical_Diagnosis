# Running the benchmark in Docker

One-command reproducibility for the MediQ compaction benchmark. The image
pins Python 3.12 and the exact dependency set the benchmark needs
(`requirements-benchmark.txt`); heavy unused deps (faiss,
sentence-transformers, neo4j) are excluded, keeping the image small.

## Build

```bash
docker build -t mediq-benchmark .
```

## Zero-cost commands (no API key needed)

```bash
# Frozen P1 plan: fingerprint, call budget, cost estimate, guards
docker run --rm mediq-benchmark

# Full offline test suite
docker run --rm --entrypoint python mediq-benchmark -m pytest \
  graphrag/eval/test_mediq_handoff_benchmark.py \
  graphrag/eval/test_mediq_handoff_holdout.py -q

# Phase 0 compression analysis (recomputes from committed checkpoints)
docker run --rm mediq-benchmark graphrag.eval.compression_analysis
```

## Paid guarded run

Needs two mounts and the API key:

- `.env` with `GOOGLE_API_KEY` (never baked into the image)
- the MedRAG textbooks BM25 index directory → `/data/mirage`
- `graphrag/eval/external` mounted read-write so the provider checkpoint and
  results persist on the host (interrupted runs resume at zero cost)

```bash
# Smoke: first 3 cases (~$0.05)
docker run --rm --env-file .env \
  -v "/Volumes/T7 Shield/RAG_Medical_Diagnosis/mirage:/data/mirage:ro" \
  -e MIRAGE_EXTERNAL_ROOT=/data/mirage \
  -v "$PWD/graphrag/eval/external:/app/graphrag/eval/external" \
  mediq-benchmark graphrag.eval.mediq_compaction_p1 --execute --limit 3

# Full n=100 run (expected $1.5-2.5, hard guard $4)
docker run --rm --env-file .env \
  -v "/Volumes/T7 Shield/RAG_Medical_Diagnosis/mirage:/data/mirage:ro" \
  -e MIRAGE_EXTERNAL_ROOT=/data/mirage \
  -v "$PWD/graphrag/eval/external:/app/graphrag/eval/external" \
  mediq-benchmark graphrag.eval.mediq_compaction_p1 --execute
```

## Guarantees carried into the container

The same guards apply as outside: frozen-spec selection is re-derived and
must match exactly, `--execute` is required for any paid call, and the
provider enforces max-calls, max-prompt-chars, and the cost guard per call.
The index is mounted read-only; the only writable surface is the checkpoint
and results directory.

## Notes

- The image was authored in a Linux sandbox without a Docker daemon; the
  dependency set is verified (the full offline test suite passes with exactly
  `requirements-benchmark.txt` on Linux/Python 3.10 and the repo venv on
  macOS/3.12+), but the first `docker build` runs on your machine.
- Code changes require an image rebuild; alternatively mount the repo over
  `/app` for development: `-v "$PWD:/app"`.
