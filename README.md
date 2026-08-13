# Does an Agent Layer Help Medical QA?

## A Controlled RAG-vs-Agent Ablation and Failure Analysis

This repository tests when Agent orchestration adds value beyond direct RAG.
The primary experiment holds the question, MedRAG Textbooks corpus, BM25
retriever, top-k 8, Gemini model, and generation instruction fixed; only the
LangGraph orchestration and validation layer changes.

The main result is negative, and that is the point:

| Finding | Evidence |
|---|---:|
| A bounded Agent did not improve forced-choice medical QA | Direct RAG 0.78 vs Agent v2 0.76, n=100; exact McNemar p=0.6875 |
| The apparent 14-point Agent improvement was removal of a harmful Reflector, not a new capability | Agent v1 0.62 → v2 0.76; the v1 Reflector regressed 11/14 initially correct drafts |
| The Agent's plausible value window—clarification and safe abstention—did not generalize cleanly | Agent v2 recall was 0/4 on development cases; Agent v3 raised external recall but clarified 0/3 and over-abstained |
| A structured Agent v3 improved safe-deferral recall but over-abstained on an external stress test | Recall 0.67 vs 0.33 Direct RAG, but false abstention 0.78 and answerable accuracy 0.22 vs 0.44 |
| The external matched-corpus evidence is BM25 Textbooks RAG, not demonstrated GraphRAG value | The project graph contains only 15 guideline chunks |

The engineering contribution is an Agent evaluation and reliability harness:
bounded tool calling, trace persistence, versioned evaluators, offline
re-scoring, selective checkpoint reuse, targeted retry, and cost guards. It is
an experiment, not a clinically validated medical device or a patient-specific
decision system.

Current architecture decision: forced-choice QA uses direct RAG plus a
deterministic answer-format validator. The Agent path is retained only as a
research prototype for open tasks that may require clarification, abstention,
tool-failure recovery, or escalation; this repository does not claim that path
is production ready.

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
| 5F | Matched-corpus RAG pilot | Complete | Textbooks RAG reached 7/10 with exact-choice parser and checkpoint reuse |
| 5G | Matched-corpus Agent pilot | Complete | Controlled RAG/Agent comparison exposed reflection/task-policy conflict |
| 5H | 100-case scaled evaluation | Complete | Agent v1 0.62 vs RAG 0.78; reflection regressed 11/14 correct drafts |
| 5I | Behavioral robustness pilot | Complete | 30 matched cases exposed zero Agent abstention recall |
| 5J | Task-policy-aware Agent | Complete | Agent v2 reached 0.76, eliminated observed reflection regressions, but did not fix abstention |
| 5K | Offline answerability shadow diagnostic | Complete | Hand-crafted development-set gate intercepted 4/4 known missing-information cases at zero API cost; no holdout claim |
| 5L | Structured answerability Agent v3 | Complete | External 36-case stress test improved safe-deferral recall but exposed severe over-abstention and task-scope mismatch |
| 5M | Provenance-aware clinical handoff scaffold | Complete (offline scaffold) | Two-question deterministic demo completes the bounded interview-to-diagnosis path; external diagnostic quality is not evaluated yet |
| 5N | MediQ clinical handoff pilot | Complete | 5-case pilot: raw patient turns 3/5, structured handoff 3/5, handoff + cited turns 4/5; n is too small for a superiority claim |

## Controlled Experimental Design

~~~mermaid
flowchart LR
    Q["Same MIRAGE question"] --> B["Same BM25 retrieval\nquestion only, top-k 8"]
    B --> C["Same MedRAG Textbooks evidence"]
    Q --> O["Answer options to generator only"]
    C --> R["Direct RAG generation"]
    O --> R
    C --> A["LangGraph Agent generation"]
    O --> A
    A --> P["Task contract + bounded reflection"]
    R --> E["Same exact-choice evaluator"]
    P --> E
~~~

The original bounded Agent runtime also implements the following enforced tool
order for the small project-guideline corpus:

~~~text
Graph -> Vector -> optional Contraindication -> Generation -> Reflection
~~~

That runtime demonstrates orchestration and safety gating, but its graph contains
only 15 chunks. The scaled external result comes from the matched Textbooks BM25
experiment above and does not establish an advantage from graph retrieval.

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

### Stage 5F — Matched-Corpus End-to-End RAG Pilot

Goal: run retrieval, context injection, Gemini generation, and exact-choice
scoring end to end on the frozen Stage 5D sample while holding the questions and
generation model fixed. This is an end-to-end RAG result; it is not yet the
LangGraph Agent comparison because `textbooks-rag` has no planning or reflection
nodes.

The prior closed-book checkpoint was imported into the new paired report, so
only the ten `textbooks-rag` calls were sent to Gemini:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 10 \
  --seed 13 \
  --systems closed-book textbooks-rag \
  --reuse-results-from graphrag/eval/external/mirage/results.json
~~~

Final parser-v3 result:

| Metric | Closed-book Gemini | Textbooks RAG |
|---|---:|---:|
| Exact-choice accuracy | 1.00 | 0.70 |
| Wilson 95% CI | [0.722, 1.000] | [0.397, 0.892] |
| Invalid-choice rate | 0.00 | 0.00 |
| Provider-error rate | 0.00 | 0.00 |
| Non-empty retrieval | N/A | 1.00 |
| Average retrieved contexts | N/A | 8.0 |
| Average retrieval latency | N/A | 0.418 s |
| p50 end-to-end latency | 2.66 s | 7.78 s |
| p95 end-to-end latency | 7.12 s | 16.88 s |
| Average total tokens | 891 | 3,550 |
| Input / output tokens | 2,293 / 6,615 | 17,247 / 18,257 |
| Estimated paid-tier cost | US$0.0172 (reused) | US$0.0508 (new) |

The cost estimate uses the recorded
[Gemini API paid-tier rates](https://ai.google.dev/gemini-api/docs/pricing) for
`gemini-2.5-flash`: US$0.30/M input tokens and US$2.50/M output tokens including
thinking, as of 2026-08-11. The report stores both the rates and raw token totals
so the estimate can be recomputed if pricing changes. The closed-book cost is
historical and was not charged again during Stage 5F; cross-report reuse saved
ten model calls.

Paired comparison:

~~~text
Textbooks-RAG wins  0
Closed-book wins    3
Ties                7
Accuracy delta     -0.30
McNemar exact p     0.25
~~~

This ten-case execution pilot is too small for a stable performance claim. The
three observed RAG failures are nevertheless useful diagnostics:

- A facial-height-ratio question retrieved general face anatomy and body-segment
  ratios, but not the requested orthodontic ratio.
- A condylar-fracture question retrieved fracture-management passages but not
  evidence for the named retromandibular transparotid approach; the generator
  incorrectly treated missing evidence as evidence for `no`.
- A Sotrovimab/COVID-19 question had no matching evidence in the older textbook
  corpus, exposing a temporal corpus-coverage gap.

The first scored run appeared to be 6/10 because parser v2 selected an incidental
`choice C` mention before a later explicit `\\boxed{A}` final answer. Parser v3
prioritizes structured JSON and boxed finals over generic mentions and re-scored
the saved raw answer to 7/10 without another Gemini call. This is why evaluator
versioning and raw-output retention are part of the evaluation system rather
than implementation details.

Two correct answers also relied on model knowledge after the retrieved passages
proved irrelevant. Therefore, answer accuracy and non-empty retrieval alone do
not establish groundedness. A scaled evaluation should add judged evidence
Recall@k or context sufficiency at retrieval, and citation/claim faithfulness at
generation, while keeping exact answer accuracy as the end-to-end outcome.

### Stage 5G — Matched-Corpus Textbooks Agent Pilot

Stage 5G injects the same Textbooks BM25 retriever into LangGraph and holds the
corpus, top-k (8), questions, seed, and Gemini model fixed. The Agent performs one
mandatory retrieval, generates an answer, and runs bounded reflection with up to
three retries. The Reflector receives up to 8,000 characters per tool context.

The Stage 5F report supplied all ten closed-book and all ten `textbooks-rag`
results, so those twenty model calls were not repeated. Only the ten
`textbooks-agent` cases were executed; a zero-token provider timeout was retried
without repeating the nine completed Agent calls:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 10 \
  --seed 13 \
  --systems closed-book textbooks-rag textbooks-agent \
  --reuse-results-from graphrag/eval/external/mirage/textbooks_results.json
~~~

Final parser-v3 result:

| Metric | Closed-book Gemini | Textbooks RAG | Textbooks Agent |
|---|---:|---:|---:|
| Exact-choice accuracy | 1.00 | 0.70 | 0.60 |
| Wilson 95% CI | [0.722, 1.000] | [0.397, 0.892] | [0.313, 0.832] |
| Invalid-choice rate | 0.00 | 0.00 | 0.20 |
| Provider-error rate | 0.00 | 0.00 | 0.00 |
| Non-empty retrieval | N/A | 1.00 | 1.00 |
| Average retrieved contexts | N/A | 8.0 | 8.0 |
| Average retrieval latency | N/A | 0.418 s | 0.404 s |
| Reflection approval rate | N/A | N/A | 0.90 |
| Average reflection retries | N/A | N/A | 0.50 |
| p50 end-to-end latency | 2.66 s | 7.78 s | 14.32 s |
| p95 end-to-end latency | 7.12 s | 16.88 s | 68.41 s |
| Average total tokens | 891 | 3,550 | 10,720 |
| Input / output tokens | 2,293 / 6,615 | 17,247 / 18,257 | 59,814 / 47,386 |
| Estimated paid-tier cost | US$0.0172 (reused) | US$0.0508 (reused) | US$0.1364 (new) |

Direct matched comparison with `textbooks-rag`:

~~~text
Textbooks-Agent wins  0
Textbooks-RAG wins    1
Ties                  9
Accuracy delta       -0.10
McNemar exact p       1.00
~~~

Against closed-book, the Agent had zero wins, four losses, six ties, an accuracy
delta of -0.40, and McNemar exact p = 0.125. This ten-case execution pilot is too
small for a stable quality claim, but it does not show a benefit from reflection:
the Agent added no correct answers over matched RAG, used about 3.0x as many
tokens, cost about 2.7x as much, and increased median latency by about 1.8x.

The two invalid choices are informative rather than parser failures. When the
textbook evidence did not cover an ovarian-torsion ligament question or the
newer Sotrovimab/COVID-19 question, reflection approved an evidence-grounded
refusal instead of the benchmark-required definite option. A third missing-
evidence case selected `maybe`, which was valid for PubMedQA but wrong. Thus the
Reflector improved caution but optimized groundedness against exact-choice task
compliance; future work should explicitly define whether insufficient evidence
requires abstention or a forced benchmark choice and score both behaviors.

### Stage 5H — 100-Case Scaled Matched-Corpus Evaluation

Goal: scale the controlled comparison to 100 cases before considering the
500-case final benchmark. The frozen seed-13 sample contains 20 cases from each
MIRAGE sub-dataset. Closed-book, `textbooks-rag`, and `textbooks-agent` use the
same questions and Gemini model; the matched-corpus systems additionally share
the MedRAG Textbooks corpus, SQLite FTS5 BM25, top-k 8, question-only retrieval,
and generation instruction. The primary variable is LangGraph orchestration and
reflection.

The 10-case Stage 5G sample is a strict subset of the 100-case sample. Stage 5H
extended checkpoint import to validate and reuse compatible subsets by pinned
benchmark hash, question-only retrieval policy, model, and system metadata. It
imported all 30 prior system-case results, representing 48 completed provider
requests, before executing the remaining 270 system-case runs. Raw generation
and judge checkpoints remain under the gitignored `graphrag/eval/external/`
directory.

Reproduce the generation run:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 100 \
  --seed 13 \
  --systems closed-book textbooks-rag textbooks-agent \
  --output graphrag/eval/external/mirage/textbooks_scaled_results.json \
  --reuse-results-from \
    graphrag/eval/external/mirage/textbooks_agent_results.json
~~~

One checkpointed judge request per case evaluates the shared top-8 evidence and
both matched-corpus outputs together. This produces evidence-sufficiency,
faithfulness, unsupported-claim, conflict-handling, and failure-slice metrics
without paying twice to judge identical retrieval:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-judge \
  --generation-report \
    graphrag/eval/external/mirage/textbooks_scaled_results.json \
  --output \
    graphrag/eval/external/mirage/textbooks_scaled_judgments.json \
  --model gemini-2.5-flash
~~~

Final parser-v4 primary result:

| Metric | Closed-book | Textbooks RAG | Textbooks Agent |
|---|---:|---:|---:|
| Exact-choice accuracy | 0.84 | 0.78 | 0.62 |
| Macro dataset accuracy | 0.84 | 0.78 | 0.62 |
| Wilson 95% CI | [0.756, 0.899] | [0.689, 0.850] | [0.522, 0.709] |
| Invalid-choice rate | 0.00 | 0.00 | 0.10 |
| Provider-error rate | 0.00 | 0.00 | 0.01 |

Per-dataset accuracy:

| Dataset | Closed-book | Textbooks RAG | Textbooks Agent |
|---|---:|---:|---:|
| MMLU | 0.90 | 0.90 | 0.70 |
| MedQA | 0.90 | 0.90 | 0.65 |
| MedMCQA | 0.80 | 0.70 | 0.45 |
| PubMedQA | 0.70 | 0.60 | 0.50 |
| BioASQ | 0.90 | 0.80 | 0.80 |

Paired exact-choice comparisons:

| Candidate vs baseline | Wins | Losses | Ties | Accuracy delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Textbooks RAG vs closed-book | 5 | 11 | 84 | -0.06 | 0.2101 |
| Textbooks Agent vs Textbooks RAG | 0 | 16 | 84 | -0.16 | 0.0000305 |
| Textbooks Agent vs closed-book | 4 | 26 | 70 | -0.22 | 0.0000595 |

The scaled result changes the interpretation of the 10-case pilot. Matched RAG
is six percentage points below closed-book, but the paired difference is not
statistically significant at this sample size. The Agent is significantly worse
than matched RAG: reflection produced no paired exact-choice wins and 16 losses.

Retrieval and judged generation metrics:

| Metric | Result |
|---|---:|
| Non-empty BM25 retrieval | 1.00 |
| Empty retrieval rate | 0.00 |
| Average retrieval latency | 0.345 s |
| Evidence sufficiency rate | 0.51 |
| Judged Recall@8 | 0.518 |
| Relevant-context rate | 0.201 |
| Outdated-corpus rate | 0.01 |
| RAG accuracy, sufficient / insufficient evidence | 0.980 / 0.571 |
| Agent accuracy, sufficient / insufficient evidence | 0.961 / 0.265 |
| RAG faithfulness / unsupported-claim rate | 0.71 / 0.25 |
| Agent faithfulness / unsupported-claim rate | 0.87 / 0.06 |
| Average unsupported claims, RAG / Agent | 0.31 / 0.06 |

Every query returned eight passages, but the judge considered only 20.1% of
those passages relevant and the complete top-8 evidence sufficient for only 51%
of cases. Non-empty retrieval therefore materially overstates retrieval quality.
When evidence was sufficient, both matched systems were highly accurate. When it
was insufficient, RAG retained 57.1% accuracy while the Agent fell to 26.5% as
reflection increasingly rejected or rewrote answers not supported by the
textbooks. No genuine evidence-conflict cases occurred, so conflict-handling is
reported as not estimable rather than as a perfect score; Stage 5I injects this
condition explicitly.

Agent orchestration metrics:

| Metric | Textbooks Agent |
|---|---:|
| Reflection approval rate | 0.97 |
| Retry-case rate | 0.24 |
| Average retries | 0.33 |
| Retry recovery | 0/7 (0.00) |
| Correct-first-draft regression | 11/14 (0.786) |
| Retrieval tool success | 0.99 |
| Tool-error rate | 0.00 |

Initial-draft tracking is available for 89 newly generated Agent cases; the ten
reused pilot cases predate that field and one case ended in a provider error.
Among traceable retries, reflection recovered none of seven initially wrong
answers and changed 11 of 14 initially correct answers into wrong or invalid
final outputs. The 0.97 approval rate is therefore not a quality metric by
itself: the Reflector often approved a grounded refusal that violated the
benchmark's required exact choice.

System performance and recorded paid-tier cost:

| Metric | Closed-book | Textbooks RAG | Textbooks Agent |
|---|---:|---:|---:|
| p50 latency | 3.48 s | 8.48 s | 15.05 s |
| p95 latency | 13.57 s | 21.40 s | 63.28 s |
| Input tokens | 16,432 | 170,982 | 558,732 |
| Output tokens | 93,553 | 198,270 | 411,838 |
| Total tokens | 109,985 | 369,252 | 970,570 |
| Logical provider requests | 100 | 100 | 260 |
| Equivalent cost | US$0.2388 | US$0.5470 | US$1.1972 |

The 100 combined judge calls used 332,703 input and 112,668 output tokens, cost
US$0.3815, and had p50/p95 latency of 5.56/6.42 seconds with zero judge errors.
After subtracting the reused pilot checkpoints, the recorded new Stage 5H spend
was approximately US$2.16; the full 100-case generation plus judging equivalent
cost was US$2.36. Timed-out requests returned no token usage and are not included
in that estimate.

The Agent used 2.63 times the RAG tokens, cost 2.19 times as much, increased p50
latency by 1.77 times and p95 latency by 2.96 times, yet lost 16 accuracy points.
Its benefit was more conservative grounding: faithfulness improved by 16 points
and unsupported-claim rate fell by 19 points. The next Agent change should not
add more reflection; it should distinguish "unsupported by retrieved evidence"
from "necessarily wrong," preserve a correct definite draft when task policy
requires a choice, and expose abstention as a separately scored behavior.

Failure slicing assigned every wrong/invalid answer one primary root cause:

| Failure category | Textbooks RAG | Textbooks Agent |
|---|---:|---:|
| Knowledge missing | 14 | 22 |
| Retrieval failure | 5 | 12 |
| Generation failure | 2 | 3 |
| Outdated corpus | 1 | 1 |
| **Total failures** | **22** | **38** |

One Agent case repeatedly returned Gemini 504 and remains a generation/provider
failure, preserving a measured 1% provider-error rate instead of retrying the
reliability signal away. A second evaluator bug was found without new model
calls: parser v3 rejected explicit Markdown-formatted finals such as
`` `answer_choice`: "B" ``. Parser v4 accepts formatting around the final-answer
key while retaining v3's protection against incidental choices in reasoning,
then re-scores saved raw outputs. This changed RAG from an apparent 0.66 with
15% invalid choices to 0.78 with no invalid choices; Agent changed from 0.60 to
0.62. Raw-output retention and evaluator versioning again prevented extra API
cost and a false system conclusion.

Stage 5H is stable enough to diagnose the current design, but its result argues
against paying for 500 cases with the unchanged Reflector. Stage 5I should first
measure robustness and abstention policy explicitly, then a revised Agent can be
re-evaluated on this frozen 100-case checkpoint before a final 500-case run.

### Stage 5I — Matched-Corpus Agent Robustness Pilot

Goal: evaluate controlled behavioral changes rather than another aggregate QA
score. The primary experiment compares a direct RAG pipeline with the LangGraph
Agent while holding the question, project guideline corpus, frozen FAISS
retrieval, top-k 8, Gemini model, and generation instruction fixed. The main
variable is Agent orchestration and reflection.

The pilot reuses the five Stage 5B families and creates no new model-generated
test data. It selects the original, paraphrase, irrelevant distractor, and
directional-or-abstention cases, then deterministically adds one conflicting-
evidence case and one retrieval-tool-failure case per family. Conflicts reuse an
existing family `forbidden_claim` inside an explicitly labeled untrusted
passage. The final 30-case matrix is:

| Slice | Cases |
|---|---:|
| Original | 5 |
| Paraphrase invariance | 5 |
| Irrelevant distractor | 5 |
| Directional/counterfactual | 1 |
| Missing evidence / abstention | 4 |
| Conflicting evidence | 5 |
| Retrieval-tool failure | 5 |
| **Total** | **30** |

Each case runs through both matched systems, and one combined judge request
evaluates both outputs. Generation and judgment are checkpointed after every
case. The command includes separate US$0.90 generation and US$0.35 judge cost
guards; local raw outputs remain under gitignored `graphrag/eval/external/`.

~~~bash
# Zero-call plan and cost estimate
.venv.nosync/bin/python -m graphrag.main robustness-benchmark --dry-run

# Primary matched project-guideline run
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main \
  robustness-benchmark \
  --corpus project-guidelines \
  --top-k 8 \
  --generation-cost-guard 0.90 \
  --judge-cost-guard 0.35
~~~

Primary behavioral result:

| Metric | Matched RAG | Matched Agent |
|---|---:|---:|
| Original reference accuracy | 3/5 (0.60) | 2/5 (0.40) |
| Overall reference accuracy | 0.467 | 0.433 |
| Overall behavior pass | 0.567 | 0.567 |
| Paired paraphrase+distractor consistency | 1.00 | 0.80 |
| Correctness retention on eligible original-correct families | 4/6 (0.667) | 4/4 (1.00) |
| Directional consistency | 1/1 (1.00) | 1/1 (1.00) |
| Abstention precision / recall / F1 | 1.00 / 0.25 / 0.40 | N/A / 0.00 / 0.00 |
| Conflict disclosure and handling | 3/5 (0.60) | 4/5 (0.80) |
| Tool-failure recovery | 5/5 (1.00) | 5/5 (1.00) |
| Unsupported-claim rate | 0.067 | 0.100 |
| Worst-slice behavior pass | 0.25 | 0.00 |
| Provider-error rate | 0.00 | 0.00 |

Behavior pass by perturbation:

| Slice | Matched RAG | Matched Agent |
|---|---:|---:|
| Paraphrase | 2/5 | 2/5 |
| Irrelevant distractor | 2/5 | 3/5 |
| Directional/counterfactual | 1/1 | 1/1 |
| Missing evidence / abstention | 1/4 | 0/4 |
| Conflicting evidence | 3/5 | 4/5 |
| Retrieval-tool failure | 5/5 | 5/5 |

The paired comparison has two Agent wins, two RAG wins, 26 ties, and exact
McNemar p = 1.0. The pilot therefore shows no net Agent quality improvement.
The Agent handled one more distractor and one more explicit conflict, but lost
one original answer and every missing-evidence abstention case. Its zero
abstention recall makes abstention the worst robustness slice, despite the
generation instruction explicitly requiring it when decisive patient facts are
missing.

System trade-offs:

| Metric | Matched RAG | Matched Agent |
|---|---:|---:|
| p50 latency | 2.25 s | 5.63 s |
| p95 latency | 5.44 s | 9.32 s |
| Input tokens | 46,034 | 91,668 |
| Output tokens | 15,187 | 35,224 |
| Total tokens | 61,221 | 126,892 |
| Provider requests | 30 | 62 |
| Estimated generation cost | US$0.0518 | US$0.1156 |
| Reflection approval | N/A | 1.00 |
| Retry-case rate | N/A | 0.033 |

The Agent used 2.07x the tokens and 2.23x the generation cost, increased p50
latency by 2.51x and p95 by 1.71x, and did not improve aggregate behavior pass.
Reflection approved every final output. Its single retry occurred on an injected
tool failure: the first draft already safely disclosed the failure, but the
Reflector incorrectly requested a definite vaccine regimen before approving a
longer version of the same abstention. This is an unnecessary-retry failure, not
a recovered medical answer.

Before the primary run, the same matrix was executed against MedRAG Textbooks
for continuity with Stage 5H. That diagnostic found zero correct original
families for either system because the internal references contain newer,
project-specific guidance such as PCV21 and rapid inpatient methadone titration
that is absent from or contradicted by the older textbooks. It produced a 0.233
behavior-pass rate for both systems, no eligible correctness-retention pairs,
and worst-slice pass of zero. Those results are retained as a corpus-mismatch
diagnostic, not presented as the Stage 5I primary robustness score. Switching to
the source-matched project guideline corpus raised behavior pass to 0.567 and
made correctness retention estimable.

The evaluator also exposed two reliability problems. First, the schema allowed
cross-field contradictions such as `abstained=true` together with
`answer_correct=true`, or a preserving `behavior_pass=true` when
`answer_correct=false`. Judge schema v4 preserves the raw fields and then
recomputes behavior from deterministic per-slice contracts. Second, the judge
sometimes passed a conflict without explicit disclosure and failed another
output that literally said the untrusted claim conflicted with trusted
evidence. Saved answer text now deterministically verifies observable conflict
and tool-failure disclosure markers. These checkpoint migrations required zero
new model calls. A remaining evaluator limitation is reference strictness: the
DKA answers explain normal/mild glucose and SGLT2 use but do not use the exact
phrase `Normoglycemic DKA`; their scores are not manually overridden without
independent clinical review.

The primary generation plus judge equivalent cost was US$0.2720. Including the
Textbooks corpus-mismatch diagnostic, total new Stage 5I spend was approximately
US$0.6475. All 120 system-case generations and 60 combined judgments completed
without provider error, and no successful paid checkpoint was repeated during
the offline evaluator migrations.

This five-family pilot is a design diagnostic, not a clinical robustness claim.
In particular, the directional slice has n=1 and the references remain marked
for human review. Before expanding to 100 robustness families or a 500-case
external benchmark, the next Agent revision should add an explicit task-policy
decision for answer versus abstention, validate that decision separately from
grounding, and avoid rewriting an already safe draft merely to satisfy the
Reflector.

### Stage 5J — Task-Policy-Aware Agent Reflection

Goal: fix the Stage 5H failure mode in which a generic grounding Reflector
overrode the benchmark contract, while testing whether a stronger open-clinical
policy could also recognize missing patient-specific information. The retrieval
experiment remains matched: Textbooks RAG and Agent v2 use the same frozen 100
questions, MedRAG Textbooks corpus, SQLite FTS5 BM25, top-k 8, question-only
retrieval, Gemini model, and generation instruction. LangGraph orchestration and
validation policy remain the main Agent variable.

The Agent state now carries an explicit per-example task mode, task policy, and
allowed answer choices. Validation applies deterministic contracts before an
LLM critique:

- A multiple-choice draft with one explicit valid choice is preserved. Weak
  retrieval cannot trigger a rewrite or abstention.
- A multiple-choice draft without a valid choice is deterministically retried;
  an LLM Reflector cannot waive the forced-choice contract.
- An open clinical answer that safely discloses an injected tool failure can be
  approved without paying for another model critique.
- Other open answers use a policy-aware prompt that asks the Reflector to
  distinguish missing retrieval evidence from decisive facts missing in the
  patient's question.

Provider requests, LLM reflection requests, and deterministic policy approvals
are counted separately. Selective checkpoint import can reuse only unchanged
systems, merge the 10-case pilot into the scaled run, and re-run a named Agent
case without repeating the other 99. The judge stores a generation fingerprint
per case, so a changed output invalidates only its own judgment. Generation and
judge cost guards stop new work if recorded token cost exceeds the configured
limit.

Reproduce the scaled generation while reusing all completed closed-book, RAG,
and 10-case Agent v2 results:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-benchmark \
  --limit 100 \
  --seed 13 \
  --systems closed-book textbooks-rag textbooks-agent \
  --output graphrag/eval/external/mirage/stage5j_agent_v2_scaled.json \
  --reuse-results-from \
    graphrag/eval/external/mirage/textbooks_scaled_results.json \
  --reuse-systems closed-book textbooks-rag \
  --additional-reuse-results-from \
    graphrag/eval/external/mirage/stage5j_agent_v2_pilot.json \
  --additional-reuse-systems textbooks-agent \
  --generation-cost-guard 1.50

PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main mirage-judge \
  --generation-report \
    graphrag/eval/external/mirage/stage5j_agent_v2_scaled.json \
  --output \
    graphrag/eval/external/mirage/stage5j_agent_v2_judgments.json \
  --cost-guard 0.70
~~~

#### Scaled MIRAGE result

Agent v2 recovered the large task-compliance loss from Agent v1 without
surpassing direct RAG:

| Metric | Closed-book | Textbooks RAG | Agent v1 | Agent v2 |
|---|---:|---:|---:|---:|
| Exact-choice accuracy | 0.84 | 0.78 | 0.62 | 0.76 |
| Macro dataset accuracy | 0.84 | 0.78 | 0.62 | 0.76 |
| Wilson 95% CI | [0.756, 0.899] | [0.689, 0.850] | [0.522, 0.709] | [0.668, 0.833] |
| Invalid-choice rate | 0.00 | 0.00 | 0.10 | 0.02 |
| Provider-error rate | 0.00 | 0.00 | 0.01 | 0.02 |

Agent v2 per-dataset accuracy was 0.90 MMLU, 0.85 MedQA, 0.65 MedMCQA,
0.55 PubMedQA, and 0.85 BioASQ. Against Agent v1 on the same cases, v2 had 14
wins, zero losses, and 86 ties: +0.14 accuracy with exact McNemar p = 0.000122.
Against Textbooks RAG, v2 had two wins, four losses, and 94 ties: -0.02 accuracy
with exact McNemar p = 0.6875. The result supports the reflection fix but does
not establish an Agent quality advantage over direct RAG.

The remaining 2% invalid-choice rate consists entirely of two provider errors.
The same MedQA and MedMCQA cases returned Gemini `504 DEADLINE_EXCEEDED` at 30,
60, and 120-second request limits, so they remain measured reliability failures
rather than being retried indefinitely. One additional MedMCQA question refers
to a missing image. Agent v2 initially abstained; the deterministic contract
forced a valid choice on retry, reducing invalid outputs without making the
arbitrary answer correct. Task compliance cannot recover absent question data.

Agent orchestration and performance changed substantially:

| Metric | Agent v1 | Agent v2 | Textbooks RAG |
|---|---:|---:|---:|
| Policy approval rate | N/A | 0.98 | N/A |
| Recorded LLM validation requests | 130 | 2 | 0 |
| Retry-case rate | 0.24 | 0.03 | N/A |
| Retry recovery | 0/7 | 1/3 | N/A |
| Correct-first-draft regression | 11/14 | 0/0 | N/A |
| p50 latency | 15.05 s | 6.77 s | 8.48 s |
| p95 latency | 63.28 s | 21.34 s | 21.40 s |
| Total tokens | 970,570 | 345,740 | 369,252 |
| Provider requests | 260 | 103 | 100 |
| Estimated generation cost | US$1.1972 | US$0.4473 | US$0.5470 |

Relative to Agent v1, v2 reduced tokens by 64%, estimated generation cost by
63%, p50 latency by 55%, and p95 latency by 66%, while gaining 14 accuracy
points. Valid choices no longer incurred a second provider call. Agent v2's
recorded token cost was slightly lower than direct RAG because its output token
total was lower, although it made three more provider requests and lost two
accuracy points. RAG latency comes from the reused Stage 5H checkpoint, so the
small recorded latency difference is not treated as a causal speed advantage
across execution dates. Two imported pilot cases used the earlier policy-aware
LLM critique before invalid-choice retry became fully deterministic; their
recorded reflection requests are retained rather than rewritten offline.

The combined evidence judge found top-8 evidence sufficient in 0.50 of cases,
judged Recall@8 of 0.508, and relevant-context rate of 0.196. RAG versus Agent v2
faithfulness was 0.74 versus 0.73; unsupported-claim rate was 0.23 versus 0.20.
Accuracy with sufficient evidence was 0.98 versus 0.96, and with insufficient
evidence 0.58 versus 0.56. Compared with Agent v1's more conservative judged
behavior, v2 restores forced-choice accuracy at the cost of no longer producing
a large faithfulness advantage. This is the intended quality/caution trade-off,
not evidence that reflection is universally beneficial. Because the changed
Agent required a new combined judge call, the unchanged RAG outputs were also
rejudged; small differences from the Stage 5H RAG judge metrics are evaluator
variation, not a new RAG generation result.

#### Open-clinical robustness result

The same 30 Stage 5I cases were re-run only for Agent v2; all 30 matched RAG
outputs were imported unchanged. A new combined judge run evaluated the reused
RAG and new Agent outputs:

~~~bash
PYTHONUNBUFFERED=1 .venv.nosync/bin/python -m graphrag.main \
  robustness-benchmark \
  --corpus project-guidelines \
  --output graphrag/eval/external/robustness/stage5j_results.json \
  --judgments graphrag/eval/external/robustness/stage5j_judgments.json \
  --reuse-results-from \
    graphrag/eval/external/robustness/stage5i_guidelines_results.json \
  --reuse-systems matched-rag \
  --generation-cost-guard 0.35 \
  --judge-cost-guard 0.20
~~~

| Metric | Reused matched RAG | Agent v2 |
|---|---:|---:|
| Overall behavior pass | 0.567 | 0.500 |
| Overall reference accuracy | 0.433 | 0.400 |
| Paired paraphrase+distractor consistency | 1.00 | 1.00 |
| Abstention recall / F1 | 0.25 / 0.40 | 0.00 / 0.00 |
| Conflict handling | 0.60 | 0.80 |
| Tool-failure recovery | 1.00 | 1.00 |
| Unsupported-claim rate | 0.10 | 0.167 |
| Worst-slice behavior pass | 0.25 | 0.00 |
| p50 / p95 latency | 2.25 / 5.44 s | 5.92 / 13.41 s |
| Total tokens | 61,221 | 136,073 |
| Provider requests | 30 | 64 |
| Estimated generation cost | US$0.0518 | US$0.1171 |

Agent v2 had zero paired behavior wins, two losses, and 28 ties against RAG
(exact McNemar p = 0.5). Most importantly, the stronger natural-language
Reflector still approved generic guideline answers in all four cases where the
patient question omitted a decisive fact: initiation protocol, vaccination
history, treatment days, or family history. Abstention recall therefore
remained zero. This isolates the root cause beyond retrieval: relevant generic
evidence was available, but generation and reflection treated population-level
guidance as sufficient for a patient-specific decision.

The RAG generation outputs are byte-for-byte reused, but some RAG reference and
unsupported-claim labels differ from Stage 5I because the combined LLM judge was
run again alongside the new Agent outputs. At n=30, these judge-only changes are
reported as evaluator variability; deterministic slice guards and exact-choice
metrics remain the primary protection against silently attributing them to the
system.

The task-aware prompt did improve one PSA answer by asking it to disclose that
the retrieved guideline lacked race-specific guidance, but it did not reliably
detect the deliberately removed family-history fact. Free-form self-reflection
is therefore not a dependable answerability classifier. The next design should
use a separate, structured query-completeness decision with explicit required
fields or independently validated answerability labels before generation; it
should not infer abstention solely from retrieval confidence.

Post-run trace inspection found that safe tool-failure drafts commonly used the
passive phrase `cannot be answered safely`, which the deterministic precheck did
not initially recognize. The final code covers both active and passive safe
abstention forms; all five saved first drafts pass the updated offline precheck.
The recorded Stage 5J latency, tokens, and cost are not retroactively changed.
This would have avoided seven provider requests in a future identical run,
including one unnecessary retry that expanded an already safe failure notice.

Stage 5J generation and judging used an estimated US$1.0507 in newly recorded
tokens: US$0.4473 MIRAGE Agent generation, US$0.3811 MIRAGE judging, US$0.1171
robustness Agent generation, and US$0.1052 robustness judging. Timed-out calls
returned no usage metadata and are excluded from the token-based estimate.
Every compatible prior closed-book, RAG, and Agent pilot generation checkpoint
was reused. Within each new judge run, successful case judgments were reused on
timeout or changed-case retries; raw paid outputs remain gitignored.

### Stage 5K — Offline Answerability Shadow Diagnostic

Goal: test one narrow hypothesis from the Stage 5I/5J failure analysis without
another model call: would an explicit query-completeness contract intercept the
four known cases where free-form reflection answered despite a missing decisive
fact?

Implemented:

- Five hand-crafted decision contracts in
  `graphrag/eval/specs/answerability_decisions.yaml`, one for each existing
  robustness family.
- Deterministic extraction of present and missing required fields before
  generation.
- A structured decision containing `decision_type`, `decision_requested`,
  `known_patient_facts`, `required_patient_facts`, `missing_required_facts`,
  `evidence_applicable`, `risk_level`, and `action`.
- `answer`, `clarify`, and `abstain` routes plus deterministic clarification
  text; unsupported decision types fail closed.
- Fingerprint validation and offline reuse of the saved Stage 5J generation and
  judgment checkpoints. No Gemini generation, retrieval, or judge request is
  made.

The rules were intentionally kept simple and auditable: initial methadone dose
requires the initiation protocol, pneumococcal regimen requires vaccination
history, target methadone dose requires the treatment window, PSA timing
requires first-degree family history under this benchmark contract, and the DKA
diagnosis requires the stated acid-base and ketone findings. These requirements
were written after seeing the five development families. They are benchmark
rules, not a general clinical ontology.

Offline development-set result:

| Metric | Saved Agent v2 | Hand-crafted shadow gate |
|---|---:|---:|
| Query-completeness cases | 25 | 25 |
| Missing-information positives | 4 | 4 |
| True positive / false positive / false negative | 0 / 0 / 4 | 4 / 0 / 0 |
| Abstention precision | Not estimable | 1.00 |
| Abstention recall | 0.00 | 1.00 |
| Abstention F1 | 0.00 | 1.00 |
| False abstention on 21 answerable controls | 0/21 | 0/21 |
| Safe action on injected retrieval failure | 5/5 | 5/5 |
| New model calls / estimated API cost | 0 / US$0.00 | 0 / US$0.00 |

The gate covers all 30 cases, but query-completeness precision and recall use the
same 25-case contract as Stage 5J: four abstention positives and 21 answerable
negative controls. The five injected tool failures are scored separately. For
ordinary answer routes, `evidence_applicable` remains `null` because a
question-only gate cannot honestly determine whether retrieved evidence applies.

This 4/4 result is a shadow diagnostic, not an accuracy improvement or a
generalization claim. The rules were designed after inspecting these cases, the
four positive labels are not clinician reviewed, the LLM judge is not calibrated,
and there is no untouched holdout. A valid extension would freeze these rules,
obtain independently reviewed cases, and evaluate once on a new holdout; this
repository stops before making that stronger claim.

Reproduce with zero external calls:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m graphrag.main \
  answerability-shadow
~~~

### Stage 5L — Structured Answerability Agent v3

Goal: freeze the Stage 5K hypothesis and test a general structured policy on an
external holdout instead of tuning another prompt on the same four known
missing-information cases. This stage evaluates answerability orchestration,
not retrieval recall: every system receives the same question and the same
benchmark-provided grounding documents.

Implemented:

- A separate LangGraph Agent v3 path; Agent v2 remains unchanged.
- A strict query-completeness schema followed by a strict
  evidence-applicability schema.
- A deterministic router selecting `answer`, `clarify`, `abstain`, or
  `escalate`; malformed gates and tool/generation errors fail closed.
- Citation validation that rejects evidence IDs not returned by retrieval.
- Injected query gate, retriever, evidence gate, and generator dependencies for
  isolated testing.
- Generation and judge checkpoints, selective error retry, per-system token,
  cost and latency accounting, output-token limits, and default cost guards.
- Operational safe-deferral, gate-clean safe-deferral, exact action,
  per-action/slice, paired win/loss, invalid-output, answerable accuracy, and
  selective-accuracy metrics.

The external source is the official RefusalBench-GaRAGe dataset pinned at
revision `bd78827d9eea5d9bb70a6424856e0120f265d886`. The 15.6 MB raw file has
SHA-256 `c4e0a3f8dc486cd3f69d1c9b814f0265c02e4eaba84a99df0176dedba70fb267`.
Seed 13 deterministically selects one answerable and one unanswerable variant
for each of 18 independent `Health` sources: 36 cases total, with three cases
from each of six refusal categories. The selection fingerprint is
`ac9633aa1ec9d9ca0b8fff86a494d50c8f7b4bdaa05971d6ef1a5af5a88119e3`.
The data is cross-model verified, not clinician reviewed, and the source's
`Health` slice includes policy, regulatory, finance, and research-method
questions as well as clinical material.

External stress-test results (`gemini-2.5-flash`, 36 cases):

| Metric | Direct RAG | Agent v2 | Agent v3 |
|---|---:|---:|---:|
| Exact action accuracy | 13/36 (0.361) | 11/36 (0.306) | 12/36 (0.333) |
| Safe-deferral precision | 0.429 | 0.462 | 0.462 |
| Safe-deferral recall | 0.333 | 0.333 | 0.667 |
| Safe-deferral F1 | 0.375 | 0.387 | 0.545 |
| False abstention on answerable cases | 0.444 | 0.389 | 0.778 |
| Clarification recall | 0/3 | 0/3 | 0/3 |
| Invalid action rate | 0.111 | 0.222 | 0.000 |
| Answerable-case accuracy after judge contract normalization | 8/18 (0.444) | 6/18 (0.333) | 4/18 (0.222) |
| Selective accuracy among emitted answers | 0.444 | 0.400 | 0.400 |

Agent v3 improved the operational recall of unsafe-to-answer cases, but it did
so by refusing too often. It did not improve exact routing or answer quality.
Its 0% invalid-action rate shows the value of a deterministic output contract,
but format reliability is not task correctness. Paired exact-action wins were
5 Direct RAG vs 4 Agent v3, with 27 ties; this small stress test does not support
a superiority claim.

System trade-offs:

| Metric | Direct RAG | Agent v2 | Agent v3 |
|---|---:|---:|---:|
| p50 / p95 latency, seconds | 5.27 / 9.32 | 8.90 / 29.21 | 9.67 / 16.96 |
| Input / output / total tokens | 55,803 / 43,004 / 98,807 | 162,736 / 95,493 / 258,229 | 50,345 / 65,860 / 116,205 |
| Recorded logical provider requests | 36 | 90 | 64 |
| Estimated generation cost | US$0.1243 | US$0.2876 | US$0.1798 |
| Extra orchestration calls | None | 45 reflections; 10 retries | 36 query gates; 18 evidence gates; 10 generations |

Agent v3 used 18% more tokens and cost 45% more than Direct RAG, while Agent v2
used 2.6 times the tokens and 2.3 times the cost. Agent v3 removed v2's long
retry tail but still increased median latency by 83% over Direct RAG. Across
generation, initial judging, and seven checkpointed judge retries, the run
recorded 215 logical requests, 522,863 tokens, and an estimated US$0.6726.

Failure analysis:

- The principal root cause was a task-contract mismatch. The query gate was
  designed for patient-specific clinical decisions, while this external slice
  contains broader health-adjacent questions. It labeled 11/36 queries
  `unsupported_task`, including 7/18 answerable controls.
- All three ambiguity cases missed the required `clarify` action. Two query
  gate responses were truncated/malformed and one violated a cross-field
  invariant, so the deterministic router correctly failed closed to `abstain`
  but could not recover the intended action.
- Four evidence-gate responses were invalid. Counting fail-closed outputs as
  operational deferrals raises Agent v3 safe-deferral F1 to 0.545; excluding
  gate errors gives 0.485 on 29 cases, with a 0.733 false-abstention rate.
- Agent v3 reached 3/3 on false premises and 2/3 on missing context, but 0/3 on
  granularity mismatch, 0/3 on clarification, and only 1/3 exact escalation on
  conflicting context.
- The first judge pass truncated 7/18 outputs at the 2,048-token limit because
  thinking consumed the response budget. The evaluator now disables judge
  thinking, permits 4,096 output tokens, stores raw output, and checkpointed
  retry reran only those seven cases. No generation was repeated.
- The judge also marked two non-`answer` actions semantically correct. A
  deterministic post-condition now forces every abstain, clarify, escalate, or
  invalid action to incorrect on answerable cases; no paid re-judging was
  required.

The Agent/gates were not tuned or rerun after inspecting this holdout. The
result argues against SFT or RL at this point: the immediate problems are task
definition, structured-output reliability, and evaluation scope. A credible
next experiment needs a frozen task taxonomy and an independently
clinician-reviewed, patient-specific holdout before considering model training.

Prepare and inspect the holdout without model calls:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m graphrag.main \
  refusalbench-holdout --download
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m graphrag.main \
  refusalbench-benchmark --dry-run
~~~

Run or resume the paid benchmark only after reviewing the dry-run budget:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m graphrag.main \
  refusalbench-benchmark
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m graphrag.main \
  refusalbench-benchmark --judge-only
~~~

Raw source data, selections, generation results, and judge outputs stay under
the gitignored `graphrag/eval/external/` directory.

### Stage 5M — Provenance-Aware Clinical Handoff Scaffold

Goal: move beyond forced-choice QA and isolate a more credible Agent capability:
bounded information gathering followed by an auditable handoff to a fresh
diagnostic context. This is an offline engineering scaffold, not a clinical
accuracy result.

Implemented:

- A bounded LangGraph loop with at most three patient questions.
- One structured interviewer update per round that both refreshes the clinical
  handoff and chooses `ask` or `finalize`. Combining these operations avoids a
  second model call per interview round.
- Patient-interaction and evidence-retrieval tools behind injected interfaces.
- Patient facts with stable fact IDs, present/absent/unknown status, and source
  patient-turn IDs; unknown or invented turn references fail closed.
- A `DiagnosticPacket` as the fresh diagnostician's only input: structured
  handoff, retrieved evidence IDs, and optionally the cited raw patient turns.
- Deterministic citation checks before the safety validator. Drafts citing facts
  or evidence that were never observed are rejected without exposing the draft.
- Explicit `answer`, `abstain`, and `escalate` outcomes, tool-call counters,
  compact traces, and fail-closed patient/retrieval/model error paths.
- An ablation switch that removes raw source turns while retaining the same
  structured handoff, enabling a future provenance-value experiment.

The zero-call demo asks two targeted questions, produces three sourced patient
facts, retrieves one fixture passage, builds a fresh diagnostic packet, and
passes its grounded fixture draft through the safety gate. It establishes
control flow and contracts only; all medical content in the demo is scripted.

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m \
  graphrag.agent.clinical_demo
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m unittest \
  graphrag.agent.test_clinical_agent -v
~~~

No provider call or estimated API cost was incurred. The next valid experiment
must connect frozen model adapters and compare raw conversation, free-text
summary, structured handoff, and structured handoff plus source turns on the
same held-out cases. Until that comparison is run, this repository makes no
claim that handoff improves diagnosis.

Validation: 118/118 active Agent, retrieval, graph, and evaluation tests passed.

### Stage 5N — MediQ Clinical Handoff Pilot

Goal: test the Stage 5M architecture on external interactive cases and isolate
the effect of the final handoff representation. This is a five-case engineering
pilot, not a clinical or statistical validation.

The official [MediQ repository](https://github.com/stellalisy/mediQ) is pinned at
revision `faa2ce62fef0423e35af4c31d7537aad973173eb`; the CC-BY-4.0 development
file has SHA-256
`3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526`.
Seed 13 selects one case from each of five high-frequency specialties, producing
source IDs `487, 586, 423, 843, 502` and dataset fingerprint
`6b6be9fdcf68601e`. The committed spec contains identifiers and source metadata,
while questions and raw outputs remain gitignored.

Controlled design:

- The same interviewer, maximum three patient questions, conversation,
  structured handoff, MedRAG Textbooks BM25 retriever, top-k 8, and
  `gemini-2.5-flash` model are reused across conditions.
- The local deterministic patient tool selects benchmark facts without model
  calls. It approximates MediQ's fact-select patient cheaply but is less capable
  than a natural-language patient simulator.
- Retrieval uses the benchmark question plus acquired patient facts and never
  answer-option text. All five cases returned eight contexts.
- The fresh diagnostic call sees either all raw patient utterances, the
  structured handoff, or the handoff plus its cited raw patient utterances.
  The first condition does not include interviewer questions; it should be
  described as a raw-patient-turn baseline, not a complete transcript baseline.
- Structured outputs, patient/evidence IDs, call count, prompt length, token
  limits, and cost are validated deterministically. Clinical traces are forced
  to stay local even if general LangSmith tracing is enabled.
- MediQ permits a forced benchmark choice with incomplete information. That
  behavior requires an explicit benchmark-only override; the clinical default
  still fails closed.

Results:

| Final diagnostic input | Exact choice | Wilson 95% CI | Valid output | Pipeline p50 / p95 | Conceptual pipeline tokens | Conceptual cost |
|---|---:|---:|---:|---:|---:|---:|
| Raw patient turns | 3/5 (0.60) | 0.23–0.88 | 4/5 (0.80) | 13.06 / 13.85 s | 35,566 | US$0.0345 |
| Structured handoff | 3/5 (0.60) | 0.23–0.88 | 5/5 (1.00) | 13.41 / 13.85 s | 38,705 | US$0.0358 |
| Handoff + cited patient turns | 4/5 (0.80) | 0.38–0.96 | 5/5 (1.00) | 13.50 / 14.32 s | 39,764 | US$0.0361 |

The interview was shared across the three diagnostic ablations, so actual pilot
cost is lower than the sum of the conceptual per-condition costs: 33 logical
provider calls, 51,177 input tokens, 13,926 output tokens, 65,103 total tokens,
and estimated cost US$0.0502. Interviewing used 18 calls and US$0.0282, 56% of
the pilot cost. Mean patient questions were 2.6; three of five cases exhausted
the question budget. Retrieval averaged 0.19 seconds and never returned empty.

Handoff + cited turns had one paired win, zero losses, and four ties against
each other representation; exact McNemar p=1.0. The apparent 20-point gain is
therefore one case and is not statistically distinguishable from chance.
Structured handoff alone changed format reliability, not exact-choice accuracy.
The heuristic lexical handoff audit measured mean revealed-fact recall 0.865
and precision 0.547; the latter is sensitive to fact granularity and unknown
statements and must not be interpreted as a clinical hallucination rate.

Failure analysis:

- On source 502, the interviewer spent all three questions on sexual activity,
  IUD use, and douching but never asked about diabetes, the decisive gold factor.
  All three diagnostic representations were wrong. Active question selection,
  not final summarization, was the primary failure.
- On source 586, the lexical patient tool could not map generic questions about
  injuries and vital signs to the corresponding hidden facts. This is a tool
  semantic-matching failure and makes the single paired handoff win fragile.
- One raw-patient-turn answer selected the correct choice but cited no textbook
  evidence; exact-choice scoring counted it correct while the contract marked
  it invalid. Structured conditions had no invalid citations.
- One interviewer output claimed readiness while retaining missing fields. A
  deterministic post-condition normalized readiness downward and preserved the
  missing-information list; no provider response was regenerated.
- Checkpoint recovery reused every completed response after two interrupted
  runs. Only one genuinely missing condition call was added at the end.

Reproduce the zero-call plan, then run or resume the guarded pilot:

~~~bash
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -m \
  graphrag.eval.mediq_handoff_benchmark --download --dry-run
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -u -m \
  graphrag.eval.mediq_handoff_benchmark --cost-guard 0.25
PYTHONDONTWRITEBYTECODE=1 .venv.nosync/bin/python -u -m \
  graphrag.eval.mediq_handoff_benchmark --reuse-only
~~~

The final command is the strict zero-call replay: it fails before any provider
request if an expected prompt is absent from the local checkpoint.

The next experiment must be frozen before looking at new cases: repair the
patient tool's medical concept mapping, use a true full-transcript baseline,
and test an untouched larger MediQ selection. The current five cases must remain
development evidence and must not be rerun after prompt tuning.

Validation: 129/129 active Agent, retrieval, graph, and evaluation tests passed.

The most consequential Agent failure was not a missing framework feature; it
was a validator optimizing the wrong objective:

~~~mermaid
flowchart LR
    D["14 initially correct Agent v1 drafts\nsent through retry/reflection"] --> W["11 became wrong or invalid"]
    D --> K["3 remained correct"]
    V2["Agent v2 deterministic\nforced-choice contract"] --> Z["0 observed correct-draft regressions"]
~~~

The Stage 5J gain from 0.62 to 0.76 is therefore described as removal of harmful
reflection and restoration of task compliance, not evidence that the Agent
learned a new reasoning capability.

Validation: the active Agent, retrieval, graph, and evaluation suite is run
before each stage commit; see the latest stage section for the exact count.

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
  graphrag.agent.test_answerability_agent \
  graphrag.agent.test_clinical_agent \
  graphrag.agent.test_tools \
  graphrag.agent.test_smoke \
  graphrag.retrieval.test_graph \
  graphrag.kg.test_builder \
  graphrag.eval.test_e2e_benchmark \
  graphrag.eval.test_mirage_corpus \
  graphrag.eval.test_mirage_benchmark \
  graphrag.eval.test_mirage_judge \
  graphrag.eval.test_robustness_benchmark \
  graphrag.eval.test_robustness_generate \
  graphrag.eval.test_answerability_shadow \
  graphrag.eval.test_refusalbench_holdout \
  graphrag.eval.test_refusalbench_benchmark \
  graphrag.eval.test_mediq_handoff_benchmark \
  graphrag.eval.test_synthetic_compare -v
~~~

## Repository Layout

~~~text
graphrag/
  agent/
    react_agent.py             bounded LangGraph workflow
    answerability_agent.py     structured Agent v3 gates and router
    clinical_agent.py          bounded interview-to-handoff workflow
    clinical_handoff.py        sourced fact and diagnostic packet contracts
    clinical_tools.py          injected patient and retrieval tool interfaces
    clinical_demo.py           zero-call deterministic handoff demo
    tools.py                   graph, vector, and safety tools
    smoke.py                   online execution smoke suite
  eval/
    e2e_benchmark.py           final-answer evaluation and local traces
    mirage_benchmark.py        external Medical MIRAGE adapter and evaluator
    mirage_judge.py            evidence/generation judge and failure slicer
    mirage_corpus.py           pinned Textbooks download and local BM25 index
    robustness_benchmark.py    matched RAG/Agent behavioral robustness pilot
    robustness_generate.py     behavioral robustness dataset generator
    answerability_shadow.py    zero-call query-completeness diagnostic
    refusalbench_holdout.py    pinned external answerability holdout adapter
    refusalbench_benchmark.py  checkpointed RAG/Agent v2/v3 stress test
    mediq_handoff_data.py pinned MediQ loader and deterministic patient tool
    checkpointed_gemini.py reusable Gemini checkpoint and cost guards
    mediq_handoff_benchmark.py interactive handoff experiment orchestration
    specs/answerability_decisions.yaml
                               five hand-crafted development contracts
    specs/mediq_handoff_pilot.json
                               pinned five-case interactive pilot design
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
- Stage 5I contains only five internal families, one directional case, and
  model-judged open-ended answers. It diagnoses policies and evaluator behavior;
  it does not estimate population-level clinical robustness.
- Stage 5H, 5I, and 5J results informed later policy design and are development
  evidence, not an untouched final test. Gemini executions on different dates
  also retain provider nondeterminism even at temperature zero.
- Stage 5K's five answerability contracts were hand-written after inspecting the
  same five families. Its 4/4 shadow result demonstrates coverage of known
  failures only; required facts and abstention labels need independent clinical
  review before a holdout test.
- Stage 5L's external holdout is cross-model verified rather than clinician
  reviewed. Its broad `Health` category includes nonclinical tasks, so it tests
  general answerability orchestration and task-scope handling, not clinical
  decision safety or retrieval quality.
- Agent v3 was deliberately frozen after the Stage 5L run. Its fail-closed gate
  errors and broad-task scope mismatch are reported as failures rather than
  prompt-tuned away on the holdout.
- The Neo4j graph contains only 15 project-guideline chunks. The strongest
  external result uses Textbooks BM25, so graph retrieval value has not been
  established by this repository.
- The LLM judge has not been calibrated against blinded repeated human ratings;
  open-ended quality scores must be treated as diagnostics rather than clinical
  outcome claims.
- This system must not be used as a substitute for professional medical care.
