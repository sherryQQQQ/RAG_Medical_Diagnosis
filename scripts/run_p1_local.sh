#!/bin/bash
# Launch the guarded Phase 1 run in the background and log to the repo so
# progress is visible from any mounted view. Safe to re-run: the provider
# checkpoint resumes completed calls at zero cost.
set -u
cd /Users/qianxinhui/Developer/RAG_Medical_Diagnosis || exit 1
LOG=p1_run.log
{
  echo "=== run_p1_local.sh invoked $(date) ==="
  if [ ! -x .venv.nosync/bin/python ]; then
    echo "ERROR: .venv.nosync/bin/python not found"
    exit 1
  fi
} >> "$LOG" 2>&1
nohup env MEDICAL_RAG_LANGSMITH_TRACING=false LANGSMITH_TRACING=false \
  .venv.nosync/bin/python -u -m graphrag.eval.mediq_compaction_p1 --execute \
  >> "$LOG" 2>&1 &
echo "started pid $!" >> "$LOG"
