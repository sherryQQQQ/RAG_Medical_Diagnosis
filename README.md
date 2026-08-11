# Medical QA with Agentic GraphRAG

This repository implements a grounded medical question-answering system that
combines FAISS semantic retrieval, a Neo4j medical knowledge graph, Gemini, and
a bounded LangGraph agent. It also separates retrieval evaluation from true
end-to-end answer evaluation so that each system stage is measured with metrics
that match its responsibility.

This is an engineering and evaluation project, not a clinically validated
medical device.

## Current Status

| Stage | Scope | Status | Primary evidence |
|---|---|---:|---|
| 1 | Reproducible Agent runtime | Complete | Environment check and dimension-compatible FAISS index |
| 2 | Validated Neo4j knowledge graph | Complete | 15/15 chunks imported, schema/path validation passed |
| 3 | Bounded safety-gated LangGraph Agent | Complete | 5/5 online smoke cases passed |
| 4 | End-to-end evaluation and observability | Complete | Frozen 5-case run, independent judge, local traces |
| 5A | Robustness dataset design | Complete | Versioned 550-case matrix, schema, offline dry-run |
| 5B | Candidate-generation pilot | Complete | 5 families / 25 schema-valid candidates, human review still required |
| 5C | External Medical MIRAGE adapter | Complete | 7,663 validated source cases and a frozen 500-case QOR selection |
| 5D | MIRAGE online execution pilot | Complete | 10 paired cases exposed corpus coverage and provider-timeout failure modes |
| 5E | Matched MedRAG Textbooks retrieval | Complete | 125,847 official snippets, pinned manifest, local BM25 index on external storage |

## Architecture

~~~mermaid
flowchart TD
    Q["Medical question"] --> A["LangGraph Agent state"]
    A --> G["Mandatory graph retrieval"]
    G --> N["Neo4j Aura"]
    N --> V["Mandatory vector retrieval"]
    V --> F["FAISS + all-mpnet-base-v2"]
    F --> D{"Known drug mentioned?"}
    D -- "Yes" --> S["Contraindication check"]
    S --> X["Gemini answer generation"]
    D -- "No" --> X
    X --> R["Grounding and safety reflection"]
    R -- "Retry, max 3" --> A
    R -- "Approved" --> O["Final answer"]
    O --> E["Stage 4 evaluator"]
    E --> J["Independent Gemini judge"]
    E --> T["Local JSON trace"]
    E -. "Explicit opt-in only" .-> L["LangSmith trace"]
~~~

The safety-critical retrieval order is enforced in code:

~~~text
Graph -> Vector -> optional Contraindication -> Generation -> Reflection
~~~

The LLM synthesizes and critiques answers, but it does not decide whether the
mandatory grounding steps can be skipped.

## Implementation Stages

### Stage 1 — Reproducible Runtime

Goal: make the existing Agent scaffold executable and reproducible before
changing its behavior.

Implemented:

- Added a pinned runtime in requirements-agent.txt.
- Added scripts/check_agent_environment.py for offline and online checks.
- Standardized generation on gemini-2.5-flash.
- Rebuilt the FAISS index so its 768 dimensions match
  sentence-transformers/all-mpnet-base-v2.
- Import PyTorch before FAISS on macOS/arm64 to avoid the native runtime conflict.
- Store the virtual environment in .venv.nosync.

Validation:

~~~text
Python imports             PASS
FAISS index                15 chunks, 768 dimensions
Gemini connectivity        PASS
Neo4j connectivity         PASS
Broken Python requirements 0
~~~

Reproduce:

~~~bash
python3 -m venv .venv.nosync
.venv.nosync/bin/python -m pip install -r requirements-agent.txt
.venv.nosync/bin/python scripts/check_agent_environment.py --online
~~~

### Stage 2 — Validated Medical Knowledge Graph

Goal: turn LLM-extracted medical entities into a constrained, repeatable Neo4j
knowledge graph rather than trusting arbitrary model output.

Implemented:

- Entity and relation allowlists.
- Deterministic endpoint normalization and canonical lowercase keys.
- Transactional MERGE operations with uniqueness constraints.
- SHA-256 SourceChunk markers for resumable, idempotent ingestion.
- Directed contraindication queries and partial-condition matching.
- Schema, constraint, sample-path, and retrieval validation.

Observed build:

| Item | Count |
|---|---:|
| Source chunks processed | 15 |
| Disease nodes | 57 |
| Symptom nodes | 17 |
| Treatment nodes | 88 |
| Drug nodes | 48 |
| Domain relationships | 131 |
| Uniqueness constraints | 5 |

An immediate rerun skipped all 15 chunks and made zero extraction calls.

Reproduce:

~~~bash
.venv.nosync/bin/python -m graphrag.main build-kg
.venv.nosync/bin/python -m graphrag.kg.validate
~~~

The source guideline file is intentionally not committed. Place the authorized
local input at final/guidelines.txt before rebuilding FAISS or Neo4j.

### Stage 3 — Bounded Safety-Gated Agent

Goal: execute a real end-to-end Agent instead of relying on the earlier
retrieval-only comparison.

Implemented:

- Mandatory Graph then Vector routing in LangGraph control flow.
- Conditional contraindication lookup when a known Neo4j drug is mentioned.
- Maximum of three tool calls and three reflection attempts.
- Safe tool-error disclosure and non-empty fallback behavior.
- Candidate-answer, validation-verdict, retry, and final-status state.
- Newest-evidence-first context ordering for the reflector.
- Five synthetic online execution cases.

Online smoke result:

| Metric | Result |
|---|---:|
| Pipeline pass | 5/5 |
| Keyword sanity pass | 5/5 |
| Reflection approved | 5/5 |
| Tool errors | 0 |
| Average retries | 0.0 |
| Average latency | 5.86 s |

These are execution smoke metrics, not clinical-quality scores. Detailed cases
are stored in graphrag/eval/data/agent_smoke_results.json.

Reproduce:

~~~bash
.venv.nosync/bin/python -m graphrag.agent.smoke --no-resume
~~~

### Stage 4 — End-to-End Evaluation and Observability

Goal: score the final Agent answer and the behavior of the complete workflow,
while preserving retrieval metrics as a separate diagnostic layer.

Implemented:

- Deterministic case selection with a dataset fingerprint.
- Checkpoint/resume with dataset and judge-model mismatch protection.
- A reference-and-context-aware LLM judge.
- Separate Agent latency and judge latency.
- Per-case tool sequence, errors, contexts, retries, reflection status, answer,
  judge rationale, and trace ID.
- 95% bootstrap confidence intervals for quality metrics.
- Local JSON traces by default.
- Optional LangSmith traces only when MEDICAL_RAG_LANGSMITH_TRACING=true.
- Judge failure preserves the already-paid Agent answer and trace.
- Empty Agent answers skip the judge call.
- Explicit --rejudge support reuses Agent answers when a judge model changes.

Frozen initial online selection:

~~~text
n=5
seed=13
fingerprint=36dd8bbe4a0aa6c4
case_ids=case_100, case_092, case_058, case_118, case_019
~~~

Dry-run without external calls:

~~~bash
.venv.nosync/bin/python -m graphrag.main e2e-benchmark \
  --limit 5 \
  --seed 13 \
  --dry-run
~~~

Run the frozen evaluation:

~~~bash
.venv.nosync/bin/python -m graphrag.main e2e-benchmark \
  --limit 5 \
  --seed 13 \
  --judge-model gemini-pro-latest \
  --no-resume
~~~

The checkpoint is written after every case to
graphrag/eval/data/e2e_agent_results.json.

Formal 5-case result:

| Metric | Result |
|---|---:|
| Pipeline success | 5/5 |
| Judge success | 5/5 |
| Reflection approval | 5/5 |
| Required Graph → Vector sequence | 5/5 |
| Tool-error case rate | 0.00 |
| Retry case rate | 0.20 |
| Clinical correctness | 0.95, 95% CI [0.85, 1.00] |
| Context faithfulness | 1.00 |
| Answer relevance | 1.00 |
| Completeness | 1.00 |
| Medical safety | 1.00 |
| Unsafe-answer rate | 0.00 |
| Unsupported claims per answer | 0.00 |
| Average Agent latency | 9.00 s |
| p50 / p95 Agent latency | 9.85 s / 13.81 s |
| Average Judge latency | 7.03 s |

Generation used gemini-2.5-flash and the independent judge used
gemini-pro-latest. LangSmith upload was disabled; each local case still records a
trace ID and full checkpoint. One answer scored 4/5 rather than 5/5 for clinical
correctness because it gave multiple context-supported vaccine regimens while
the reference expected one specific option.

Lexical token F1 was only 0.143 despite the strong reference-aware scores. This
is expected for open-ended answers with different valid wording and is why token
F1 is retained as a diagnostic rather than the primary medical QA metric.
Because n=5 is small, these numbers are a pipeline-level preliminary evaluation,
not a deployment or clinical-validity claim.

### Stage 5A — Robustness Dataset Design

Goal: replace a small aggregate evaluation with a versioned behavioral test
matrix that can distinguish system robustness from evaluator robustness before
paying for a large API run.

Implemented:

- Deterministically select 100 independent canonical families from the 125-row
  medical QA source dataset.
- Generate five cases per family: original, paraphrase, lay-language rewrite,
  irrelevant-noise variant, and either a directional-change or abstention case.
- Reserve 50 additional cases for prompt injection, conflicting evidence,
  Graph failure, Vector failure, and out-of-domain behavior.
- Record family/source IDs, expected behavior, gold facts, forbidden claims,
  safety severity, paired-case links, fault injection, review status, and model
  provenance for every case.
- Publish a machine-readable JSON Schema for the per-case contract.
- Validate exact quotas, unique IDs, non-empty references, preserving-pair
  invariants, source/spec fingerprints, and dataset completeness.
- Checkpoint after each generated family and reuse completed paid calls on resume.
- Define `family_id` as the statistical bootstrap unit so correlated rewrites are
  not incorrectly counted as independent evidence.

Planned dataset:

| Slice | Cases |
|---|---:|
| 100 canonical questions x 5 behavioral variants | 500 |
| Prompt injection | 15 |
| Conflicting evidence | 10 |
| Graph tool unavailable | 10 |
| Vector tool unavailable | 10 |
| Out of domain | 5 |
| **Total** | **550** |

Behavioral and adversarial quotas are hard constraints. Clinical capability
counts are soft coverage targets: the completed generator reports target gaps,
and Stage 5B must resample or review a mismatch instead of relabeling an
incompatible source question merely to make the table balance.

The source questions have been used during internal project development, so
this suite is explicitly labeled `internal_behavioral_evaluation`. It tests
metamorphic consistency and failure handling; it is not presented as an
external clinical-validation set.

Offline design validation, with zero external calls:

~~~bash
.venv.nosync/bin/python -m graphrag.main robustness-generate --dry-run
~~~

Generate and checkpoint the frozen candidate dataset (Stage 5B, external Gemini
calls; run only after reviewing a small sample and approving the cost):

~~~bash
.venv.nosync/bin/python -m graphrag.main robustness-generate --no-resume
~~~

The frozen specification is stored in
`graphrag/eval/specs/robustness_matrix.yaml`, with the formal case contract in
`graphrag/eval/specs/robustness_case.schema.json`. Stage 5A does not run the
Agent or claim robustness results; it establishes the test contract for Stage
5B onward.

### Stage 5B — Candidate-Generation Pilot

Goal: test the generator on a small paid sample before committing to the full
550-case generation run.

The preview command sends only the selected source questions and references to
Gemini. It does not execute the Agent, judge answers, or generate the 50
adversarial cases:

~~~bash
.venv.nosync/bin/python -m graphrag.main robustness-generate \
  --preview-families 5 \
  --no-resume
~~~

Final pilot result:

| Metric | Result |
|---|---:|
| Canonical families | 5 |
| Total candidate cases | 25 |
| Original cases | 5 |
| Paraphrase / lay-language / distractor cases | 5 / 5 / 5 |
| Directional / abstention cases | 1 / 4 |
| Preserving-reference checks | 15/15 |
| Duplicate normalized questions | 0 |
| Safety-critical families | 5/5 |
| Dataset fingerprint | `4f1a5e0ccd8af9ce` |

The pilot caught failures that a JSON-schema-only check would have missed:

- An abstention variant omitted required reference and critical-change fields.
- Several distractors added clinical facts such as allergy history, exercise,
  or blood pressure.
- A lay-language rewrite dropped a race attribute from the source question.
- A distractor added a date, and an insufficient-information answer was
  mislabeled as a directional test.

Changes made from those failures:

- Retry semantically invalid JSON with the exact validation error and previous
  payload while retaining per-family checkpoints.
- Require preserving variants to retain all numeric and protected demographic
  surface facts.
- Reject distractors that introduce clinical details, dates, identifiers, or
  extra numeric facts.
- Enforce consistency between directional/abstention labels and the expected
  reference behavior.

The final pilot is stored at
`graphrag/eval/data/robustness_preview.json`. Its cases remain marked for human
review: this engineering inspection verifies the test transformation contract,
not independent clinician approval of the underlying medical references. No
Agent robustness metric is reported yet.

### Stage 5C — External Medical MIRAGE Adapter

Goal: add an independently published medical RAG benchmark so that internal
robustness results are not mistaken for external generalization evidence.

The adapter targets [Medical MIRAGE](https://github.com/gzxiong/MIRAGE)
(Medical Information Retrieval-Augmented Generation Evaluation), the 7,663-case
benchmark described in the
[ACL 2024 paper](https://aclanthology.org/2024.findings-acl.372/). It validates
all five source datasets and freezes an evenly stratified 500-case first run.

Offline integration result:

| Item | Result |
|---|---:|
| Official source cases validated | 7,663 |
| Frozen evaluation cases | 500 |
| MMLU / MedQA / MedMCQA | 100 / 100 / 100 |
| PubMedQA / BioASQ | 100 / 100 |
| Selection seed | 13 |
| Selection fingerprint | `e7074121b1a13d9d` |
| Source SHA-256 | `6f7f08c64cd2efe0...` |
| Gemini or Agent calls during preparation | 0 |

Implemented:

- Download the official JSON from a pinned URL and reject a hash mismatch.
- Keep the external questions and run outputs under a gitignored local cache.
- Validate dataset counts, questions, options, and gold choices before sampling.
- Separate `retrieval_query` from `answer_query`: Graph and Vector receive only
  the question, while the generator receives the question and answer options.
  This enforces MIRAGE's question-only retrieval setting and prevents option
  leakage into retrieval.
- Compare `closed-book` and `current-agent` on identical paired cases.
- Use exact choice accuracy rather than an LLM judge, with invalid-choice rate,
  per-dataset and macro accuracy, Wilson 95% intervals, paired win/loss counts,
  exact McNemar tests, latency, token use, and Agent tool-sequence diagnostics.
- Checkpoint after every system/case call and reject resume attempts with a
  different selection, model, or system list.

Download and reproduce the zero-cost frozen selection:

~~~bash
.venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --download \
  --limit 500 \
  --seed 13 \
  --dry-run
~~~

Run a small paid execution pilot before authorizing the full selection:

~~~bash
.venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 10 \
  --seed 13 \
  --systems closed-book current-agent \
  --no-resume
~~~

The full 500-case run is intentionally not executed as part of Stage 5C because
it makes paid Gemini calls. More importantly, `current-agent` still retrieves
from this project's 15-chunk guideline corpus. Its accuracy is an honest corpus
coverage diagnostic, but it is not comparable to MIRAGE/MedRAG published RAG
scores. Integrating the official retrieved snippets or a matched benchmark
corpus is the next prerequisite for a fair retrieval-system comparison.

### Stage 5D — MIRAGE Online Execution Pilot

Goal: execute the external adapter on a small paid sample, validate exact-choice
scoring and checkpoint recovery, and identify scaling blockers before running
hundreds of cases.

The frozen pilot contains two cases from each MIRAGE sub-dataset:

~~~text
n=10
seed=13
fingerprint=4ffb0815ecb344c2
mmlu=2, medqa=2, medmcqa=2, pubmedqa=2, bioasq=2
~~~

Final parser-v2 result:

| Metric | Closed-book Gemini | Current Agent |
|---|---:|---:|
| Exact-choice accuracy | 1.00 | 0.20 |
| Wilson 95% CI | [0.722, 1.000] | [0.057, 0.510] |
| Invalid-choice rate | 0.00 | 0.60 |
| Provider-error rate | 0.00 | 0.10 |
| Graph → Vector sequence | N/A | 0.90 |
| p50 latency | 2.66 s | 14.45 s |
| p95 latency | 7.12 s | 43.32 s |
| Average total tokens | 891 | 4,649 |

Paired comparison:

~~~text
Agent wins       0
Closed-book wins 8
Ties             2
Accuracy delta  -0.80
McNemar exact p  0.0078125
~~~

The ten cases are an execution pilot, not a stable estimate of benchmark
performance. The result is nevertheless diagnostic: the current Agent enforces
grounding against its local corpus, while the closed-book model can answer from
parametric knowledge. Five Agent outputs explicitly abstained because the
15-chunk corpus contained no relevant evidence, one request ended with a Gemini
504, two answers were incorrect, and two were correct. Therefore, the low score
primarily demonstrates corpus mismatch plus one provider failure; it should not
be presented as a fair comparison with MedRAG systems using biomedical corpora.

Problems caught and changes made:

- A Gemini response stalled long enough to block visible progress. Gemini
  request timeout and retry limits are now configurable and bounded; errors are
  checkpointed so the next case can continue.
- Buffered terminal output made a progressing batch look frozen. Operational
  runs use unbuffered Python output when live progress is required.
- One correct closed-book answer used LaTeX `\\boxed{D}` instead of the requested
  JSON. Parser v2 recognizes this explicit answer form and re-scores saved raw
  outputs without repeating paid model calls.
- Two manual interruptions tested checkpoint recovery in practice: completed
  system/case pairs were reused rather than sent to Gemini again.

Reproduce or resume the pilot:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 10 \
  --seed 13 \
  --systems closed-book current-agent
~~~

The detailed report stays in the gitignored local external-data directory. The
next evaluation stage should integrate MIRAGE's official retrieved snippets or
a matched biomedical corpus before increasing the Agent run to 100 or 500
cases. Scaling the current 15-chunk corpus run would produce a more precise
measurement of a known coverage mismatch rather than a fair RAG benchmark.

### Stage 5E — Matched MedRAG Textbooks Retrieval

Goal: replace the 15-chunk corpus mismatch with a corpus used by the published
MIRAGE/MedRAG study, while keeping the laptop setup reproducible and the
external source text out of Git.

The official MIRAGE archive of top-10k IDs for every corpus/retriever/task
combination is approximately 18.9 GB compressed, before the referenced corpora
are available locally. Stage 5E instead uses the official
[MedRAG Textbooks corpus](https://huggingface.co/datasets/MedRAG/textbooks): 18
medical textbooks and 125,847 pre-chunked snippets. This is a matched MIRAGE
corpus, but the local SQLite FTS5 BM25 implementation must still be reported as
its own retriever configuration rather than as a published leaderboard run.

Implemented:

- Pin the corpus revision and all 18 file sizes/SHA-256 values in a versioned
  source manifest.
- Download each file atomically, reuse validated files, and reject size/hash
  mismatches.
- Build a persistent dependency-free SQLite FTS5 BM25 index with unique snippet
  IDs and stored corpus/index provenance.
- Keep all source text and the generated index outside the repository through
  `MIRAGE_EXTERNAL_ROOT`.
- Add a `textbooks-rag` MIRAGE system that retrieves with the question only,
  injects top-k documents into a MedRAG-style prompt, and records snippet IDs,
  context count, retrieval latency, model tokens, and exact-choice accuracy.
- Keep `textbooks-rag` separate from `current-agent`; this isolates the effect
  of fixing the corpus before dependency-injecting it into LangGraph.

Observed local build on `/Volumes/T7 Shield`:

| Item | Result |
|---|---:|
| Source files validated | 18/18 |
| Source bytes | 211,559,353 (201.8 MiB) |
| Indexed snippets | 125,847 |
| Manifest fingerprint | `fb792a392bb95287` |
| SQLite index bytes | 191,430,656 (182.6 MiB) |
| Combined disk use | 389 MiB |
| Frozen 10-case non-empty retrieval | 10/10 |
| Average / maximum retrieval latency | 141 ms / 367 ms |

Non-empty retrieval is an execution metric, not a relevance judgment. The
retrieved passages still require downstream answer-accuracy evaluation or
human relevance labels before claiming retrieval quality.

Configure the external location in the local `.env`:

~~~text
MIRAGE_EXTERNAL_ROOT=/Volumes/T7 Shield/RAG_Medical_Diagnosis/mirage
~~~

Reproduce or validate the corpus and index:

~~~bash
.venv.nosync/bin/python -m graphrag.main mirage-corpus \
  --download \
  --build-index

.venv.nosync/bin/python -m graphrag.main mirage-corpus --dry-run
~~~

Query the local index without Gemini:

~~~bash
.venv.nosync/bin/python -m graphrag.main mirage-corpus \
  --query "facial nerve compression at the stylomastoid foramen" \
  --top-k 3
~~~

The next paid pilot can compare `closed-book` with `textbooks-rag` on the same
frozen ten questions. A subsequent stage should inject this same retriever into
LangGraph so RAG generation and Agent orchestration can be compared while
holding the corpus fixed.

## Metrics by System Stage

Different stages require different metrics. Adding more metrics is useful only
when each metric diagnoses a distinct failure mode.

| Layer | Metrics | What a failure means |
|---|---|---|
| Retrieval | MRR, Top-1/3/5 | Relevant evidence was ranked too low or missed |
| Orchestration | Required tool sequence, tool-error rate, context count | Agent routing or tool integration failed |
| Reflection | Approval rate, retry rate, average retries | Drafts are repeatedly ungrounded or incomplete |
| Answer quality | Correctness, faithfulness, relevance, completeness | Final answer does not match the reference or evidence |
| Medical safety | Safety score, unsafe-answer rate, unsupported claims | Answer may create clinical risk or false reassurance |
| System performance | Pipeline/judge success, p50/p95 Agent latency | Reliability or user-facing performance is poor |
| Behavioral robustness | Invariance, directional consistency, abstention F1, recovery rate, worst slice | Behavior changes incorrectly under controlled perturbations |
| External benchmark | Exact choice accuracy, per-dataset/macro accuracy, invalid-choice rate, McNemar test | The system does not generalize to independent medical QA data |
| Evaluator reliability | Human/Judge agreement, order/verbosity sensitivity, repeatability | The metric may be unstable even when the Agent output is unchanged |

Retrieval metrics cannot establish final answer quality. Likewise, a high
answer-quality score cannot identify whether weak retrieval, routing, generation,
or reflection caused a failure. The evaluator therefore reports both stage-level
diagnostics and end-to-end outcomes.

## Retrieval-Only Benchmark

The deterministic offline benchmark contains 375 synthetic queries derived from
final/medical_generalization.csv. It does not call Gemini or execute the
LangGraph Agent.

| Method | MRR | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| Vector-only baseline | 0.598 | 0.541 | 0.648 | 0.688 |
| Hybrid graph+vector | 0.853 | 0.781 | 0.901 | 0.971 |

Reproduce:

~~~bash
.venv.nosync/bin/python -m graphrag.main generate-synthetic-data
.venv.nosync/bin/python -m graphrag.main synthetic-benchmark
~~~

Results are stored in
graphrag/eval/data/retrieval_benchmark_results.json.

## Environment

Copy the template and fill in local credentials:

~~~bash
cp .env.example .env
chmod 600 .env
~~~

Required:

~~~text
GOOGLE_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_REQUEST_TIMEOUT_S=30
GEMINI_MAX_RETRIES=1
NEO4J_URI=
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
~~~

Evaluation and observability:

~~~text
EVAL_JUDGE_MODEL=gemini-pro-latest
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=medical-graphrag
MEDICAL_RAG_LANGSMITH_TRACING=false
~~~

MEDICAL_RAG_LANGSMITH_TRACING defaults to false. Enabling it uploads prompts,
retrieved context, outputs, and trace metadata to the configured LangSmith
project.

Gemini calls use a bounded request timeout and retry count. This prevents one
stalled provider response from hanging a multi-case evaluation indefinitely;
the evaluator checkpoints the timeout as a case error and continues.

## Tests

Run the focused suite without writing Python bytecode:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m unittest \
  graphrag.agent.test_react_agent \
  graphrag.agent.test_tools \
  graphrag.agent.test_smoke \
  graphrag.retrieval.test_graph \
  graphrag.kg.test_builder \
  graphrag.eval.test_e2e_benchmark \
  graphrag.eval.test_mirage_corpus \
  graphrag.eval.test_mirage_benchmark \
  graphrag.eval.test_robustness_generate \
  graphrag.eval.test_synthetic_compare -v
~~~

## Repository Layout

~~~text
graphrag/
  agent/
    react_agent.py             bounded LangGraph workflow
    tools.py                   graph, vector, and safety tools
    smoke.py                   online execution smoke suite
  eval/
    e2e_benchmark.py           final-answer evaluation and local traces
    mirage_benchmark.py        external Medical MIRAGE adapter and evaluator
    mirage_corpus.py           pinned Textbooks download and local BM25 index
    robustness_generate.py     behavioral robustness dataset generator
    synthetic_compare.py       retrieval-only comparison
    data/                      versioned evaluation artifacts
    external/                  gitignored benchmark cache and paid run outputs
  kg/
    builder.py                 validated idempotent KG ingestion
    validate.py                Neo4j schema and path checks
  retrieval/
    graph.py                   Neo4j retrieval and safety queries
    vector.py                  FAISS semantic retrieval
scripts/
  check_agent_environment.py   reproducibility and connectivity checks
~~~

## Evaluation Limits

- The 375-query retrieval set is synthetic and is not a substitute for a
  human-labeled external retrieval benchmark.
- The Stage 4 source references come from the project dataset; a final claim
  requires a frozen human-reviewed test set.
- The Stage 5C MIRAGE adapter supplies external QA labels, but the current
  15-chunk local corpus is not the MedRAG corpus. Do not compare current-Agent
  RAG accuracy with the public leaderboard until corpus inputs are aligned.
- LLM-as-Judge scores may be biased. The evaluator records whether the judge is
  independent from the generation model and retains its rationale.
- Automated safety scores require manual review of every answer marked unsafe
  and a stratified review of answers marked safe.
- This system must not be used as a substitute for professional medical care.
