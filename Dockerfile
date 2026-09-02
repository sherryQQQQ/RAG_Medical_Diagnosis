# Reproducible runtime for the MediQ compaction benchmark.
#
# Build:
#   docker build -t mediq-benchmark .
#
# Free dry-run (prints the frozen plan, no API calls):
#   docker run --rm mediq-benchmark
#
# Paid guarded run (requires API key + the BM25 index volume; see docs/docker.md):
#   docker run --rm --env-file .env \
#     -v "/Volumes/T7 Shield/RAG_Medical_Diagnosis/mirage:/data/mirage:ro" \
#     -e MIRAGE_EXTERNAL_ROOT=/data/mirage \
#     -v "$PWD/graphrag/eval/external:/app/graphrag/eval/external" \
#     mediq-benchmark graphrag.eval.mediq_compaction_p1 --execute
#
# The provider checkpoint and results write into the mounted
# graphrag/eval/external volume, so interrupted runs resume at zero cost.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-benchmark.txt .
RUN pip install --no-cache-dir -r requirements-benchmark.txt

COPY graphrag/ graphrag/

# Tracing off by default; unbuffered so per-case cost lines stream immediately.
ENV LANGSMITH_TRACING=false \
    LANGCHAIN_TRACING_V2=false \
    MEDICAL_RAG_LANGSMITH_TRACING=false \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m"]
CMD ["graphrag.eval.mediq_compaction_p1", "--dry-run"]
