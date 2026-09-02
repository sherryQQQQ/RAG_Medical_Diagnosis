# Phase 0 — Zero-Cost Compression Analysis (Stages 5N/5O)

Produced by `python -m graphrag.eval.compression_analysis` from the committed
Stage 5N/5O checkpoints. **No model calls were made** (`model_calls_made: 0` in
`graphrag/eval/data/compression_analysis_metrics.json`). All numbers below are
recomputed from data already paid for.

Scope note: Stage 5O is the primary evidence base (n=30, untouched holdout at
the time it ran; it is now **development evidence** because it informed this
plan). Stage 5N (n=5 pilot) is reported for completeness only.

## P0.1 Compression accounting

Diagnostic-call input tokens (exact, from provider usage metadata; Stage 5O,
n=30 per condition):

| Condition | Mean | Median | Min | Max | vs full-transcript (mean) |
|---|---|---|---|---|---|
| full-transcript | 2,241 | 2,180 | 1,683 | 3,844 | 1.00× |
| structured-handoff | 2,532 | 2,586 | 1,817 | 2,945 | 1.13× |
| handoff-plus-sources | 2,824 | 2,816 | 1,965 | 4,868 | 1.26× |

Representation-level size (heuristic: the serialised patient input exactly as
`_diagnosis_prompt` builds it, characters; token estimate = chars/4):

- handoff / transcript ratio: **mean 2.14×, median 2.08×** (range 1.34–3.57)
- Stage 5N shows the same pattern (mean 2.40×).

**Honest headline: at current transcript lengths (≤3 questions), the
structured handoff is NOT a compression.** The JSON schema (fact IDs,
categories, status fields, provenance links, missing-information and
contradiction lists) costs more tokens than the short raw transcript it
replaces. The retrieval evidence block (~7.5k chars cap) dominates every
diagnostic prompt regardless of condition, so the input-token spread between
conditions is modest (+13% / +26%).

Two implications, stated without spin:

1. The current value of the handoff at these lengths is **auditability and
   accuracy** (25/30 vs 22/30 in 5O; paired analysis in the 5O report), not
   token savings.
2. Any compression claim requires longer, noisier transcripts, where the
   handoff's size is roughly fixed while the transcript grows. That is
   precisely the Phase 2 long-context stress test; a compression benefit at
   ≤3 questions should not be claimed and is not claimed here.

Interesting structural detail: structured-handoff input tokens have a much
tighter spread (max 2,945) than full-transcript (max 3,844) — the handoff
bounds the diagnostic context size, which is the mechanism the stress test
will probe.

## P0.2 Fact-preservation audit (offline, lexical heuristics)

Stage 5O (30 cases, 244 handoff facts):

| Metric | Value | Method |
|---|---|---|
| Key-fact recall (mean, lexical) | 0.874 | existing per-case lexical audit, aggregated |
| Key-fact precision (mean, lexical) | 0.607 | same; low precision = handoff includes facts beyond the revealed key-fact list (question framing, chief complaint), not fabrications |
| Negation retention | 14/15 = 0.933 | source facts containing negation cues (no/denies/without/absent/…) lexically preserved in handoff statements |
| Fabricated-fact rate | **0/244 = 0.0** | facts whose `source_turn_ids` cite a nonexistent or non-patient turn; fail-closed citation validation predicts 0 and the audit confirms it |

Stage 5N (5 cases, 45 facts): recall 0.865, precision 0.547, negation
retention 2/2, fabricated facts 0/45.

All of these are **lexical heuristics** (content-token overlap ≥ 0.5), not
semantic equivalence judgments; they can miss paraphrases and over-credit
surface overlap. Labelled as such per the honesty rules.

## P0.3 Failure catalog (Stage 5O, every case wrong in ≥1 condition)

8 of 30 cases were wrong in at least one condition. Root-cause labels are
pattern-based inferences from cross-condition correctness + per-case fact
recall, not causal proof.

| Case | Specialty | Qs | Fact recall | FT | SH | H+S | Root-cause label |
|---|---|---|---|---|---|---|---|
| mediq-1049 | Internal Medicine | 2 | 1.00 | ✗ | ✗ | ✗ | information-acquisition / intrinsic |
| mediq-141 | Internal Medicine | 3 | 0.80 | ✗ | ✗ | ✗ | information-acquisition / intrinsic |
| mediq-162 | Emergency Medicine | 2 | 1.00 | ✗ | ✗ | ✗ | information-acquisition / intrinsic |
| mediq-38 | Emergency Medicine | 3 | 0.80 | ✗ | ✗ | ✗ | information-acquisition / intrinsic |
| mediq-866 | OB/GYN | 3 | 0.83 | ✗ | ✗ | ✗ | information-acquisition / intrinsic |
| mediq-666 | Emergency Medicine | 3 | 0.67 | ✗ | ✓ | ✓ | transcript-noise candidate |
| mediq-82 | Pediatrics | 3 | 1.00 | ✗ | ✓ | ✓ | transcript-noise candidate |
| mediq-637 | Neurology | 3 | 1.00 | ✗ | ✓ | ✓ | transcript-noise candidate |

Quantified takeaways:

- **Zero handoff-loss failures.** No case was answered correctly from the full
  transcript but missed by the structured handoff. Every handoff-arm error
  (5/5) was also wrong in the transcript arm.
- **Information acquisition dominates** (5/8 failures wrong in all three
  conditions): once the interviewer fails to surface the discriminative fact
  within the 3-question budget, no downstream representation recovers it.
  This quantifies the prior observation that question selection is the
  bottleneck.
- The transcript arm additionally lost 3 cases that both handoff arms got
  right — consistent with (but at n=3 not proof of) representation noise in
  the raw transcript. Phase 2 tests this deliberately via distractor
  injection.

---

## Phase 1 — Pre-Registered Holdout (n=100, seed 47)

Produced by `python -m graphrag.eval.mediq_compaction_p1 --execute`.
Spec frozen in `graphrag/eval/specs/mediq_compaction_p1.json` (fingerprint
`9e3e9df2925823ed`) before any run. 98/100 cases completed; 2 failed with
`JSONDecodeError` (malformed model output, caught by per-case fault tolerance).
7 cases ended in `interview_error` (agent could not conduct a valid interview)
— they are excluded from the diagnosis-stage analysis. **Evaluated n = 91.**
Total cost: $1.32 (expected $1.5–2.5, hard guard $4).

## P1.1 Accuracy by condition

| Condition | Correct | N | Accuracy | 95% Wilson CI |
|---|---|---|---|---|
| full-transcript | 52 | 91 | **57.1%** | [46.9%, 66.8%] |
| truncation-headtail | 54 | 91 | **59.3%** | [49.1%, 68.9%] |
| freetext-summary | 47 | 91 | **51.6%** | [41.5%, 61.6%] |
| structured-handoff | 54 | 91 | **59.3%** | [49.1%, 68.9%] |
| handoff-plus-sources | 55 | 91 | **60.4%** | [50.2%, 69.9%] |

All 95% CIs overlap. No condition is statistically significantly different
from any other at n=91.

## P1.2 Primary comparison — McNemar paired test

Pre-registered primary: structured-handoff vs full-transcript.

| | primary wins | baseline wins | ties | McNemar exact p |
|---|---|---|---|---|
| structured-handoff vs full-transcript | 5 | 3 | 83 | **0.73** |
| structured-handoff vs truncation-headtail | 4 | 4 | 83 | 1.00 |
| structured-handoff vs freetext-summary | 11 | 4 | 76 | 0.118 |
| structured-handoff vs handoff-plus-sources | 4 | 5 | 82 | 1.00 |

**Primary result: p = 0.73. The structured handoff does NOT improve diagnostic
accuracy over the raw full transcript at n=91 pre-registered cases.**

Secondary comparisons are labelled exploratory. The handoff significantly
outperforms freetext-summary is not claimed (p=0.12, pre-alpha); that
comparison is hypothesis-generating only.

## P1.3 Compression accounting — Phase 1

Diagnosis-call input tokens (exact, from JSONL run log, n=91 per condition):

| Condition | Mean tokens/case | Total | vs full-transcript |
|---|---|---|---|
| full-transcript | 2,161 | 196,570 | 1.00× |
| truncation-headtail | 2,155 | 196,141 | **0.998×** (virtually identical) |
| freetext-summary | 1,870 | 170,148 | **0.87×** (−13%) |
| structured-handoff | 2,601 | 236,687 | **1.20×** (+20%) |
| handoff-plus-sources | 2,779 | 252,857 | **1.29×** (+29%) |

The Phase 0 finding is replicated at scale: **the structured handoff adds
diagnostic-stage tokens, it does not compress.** Truncation-headtail matches
transcript size exactly by design (budget-matched). Freetext-summary is the
only condition that reduces diagnostic input tokens, at the cost of the lowest
accuracy (51.6%).

Valid output rates (outputs passing the structured-diagnosis schema check):

| Condition | Valid outputs | Valid output rate |
|---|---|---|
| full-transcript | 72/91 | 79.1% |
| truncation-headtail | 68/91 | 74.7% |
| freetext-summary | 68/91 | 74.7% |
| structured-handoff | 72/91 | **79.1%** |
| handoff-plus-sources | 69/91 | 75.8% |

The handoff produces valid structured outputs at the same rate as the raw
transcript, while truncation and freetext-summary produce fewer valid outputs.
This is a schema-compliance benefit, not an accuracy benefit.

## P1.4 Interview-error and failure analysis

- **7 interview_error cases** (agent failed to complete a valid interview;
  these cases contributed no diagnosis-stage data): mediq-880, mediq-68,
  mediq-881, mediq-1134, mediq-902, mediq-1110, mediq-735.
- **2 failed cases** (JSONDecodeError in model output, caught per-case):
  mediq-889, mediq-437. These were recorded in `failed_cases` and excluded.
- The 7 interview errors (~7% of cases) confirm the Phase 0 bottleneck:
  question selection and interview conduct limits downstream accuracy, and
  this limitation is invisible to any representation-level comparison.

## P1.5 Honest summary

1. **No significant accuracy difference** between any arm at n=91.
2. The structured handoff ties with truncation-headtail (both 59.3%) and
   outperforms freetext-summary numerically (not significantly).
3. **Handoff does not compress** at these transcript lengths — it costs +20%
   diagnostic input tokens over the raw transcript.
4. **Schema compliance**: handoff and full-transcript are tied at 79.1% valid
   outputs; simpler conditions score lower.
5. The compression claim requires Phase 2 (longer, noisier transcripts) to
   test; that claim is not supported at ≤3 questions and is not made here.

---

## P1.6 Calibration analysis (zero-cost, offline)

Produced by `python -m graphrag.eval.calibration_analysis` from P1 results.
Full metrics: `graphrag/eval/data/calibration_metrics.json`.

**confidence=0 = invalid-schema output (model declined to produce a
valid structured diagnosis). These are "forced abstentions", not deliberate
uncertainty signals.**

### Summary table

| Arm | Acc (all) | Abstain% | Brier↓ | ECE↓ (active) |
|---|---|---|---|---|
| full-transcript | 57.1% | 20.9% | **0.336** | 0.239 |
| truncation-headtail | 59.3% | 25.3% | 0.356 | **0.207** |
| freetext-summary | 51.6% | 25.3% | 0.356 | 0.293 |
| structured-handoff | 59.3% | 24.2% | 0.384 | 0.248 |
| handoff-plus-sources | 60.4% | 25.3% | 0.377 | 0.241 |

Lower is better for Brier and ECE.

### Risk-coverage at confidence thresholds

Accuracy (coverage) when only answering cases above threshold:

| Threshold | full-transcript | truncation | freetext | structured-handoff | handoff+sources |
|---|---|---|---|---|---|
| 0.0 (all) | 57.1% (100%) | 59.3% (100%) | 51.6% (100%) | 59.3% (100%) | 60.4% (100%) |
| 0.8 | 62.7% (65%) | 66.7% (56%) | 57.6% (65%) | 64.2% (58%) | 64.7% (56%) |
| 0.9 | 75.6% (45%) | 77.8% (40%) | 66.7% (43%) | 71.4% (38%) | **81.6% (42%)** |

### Key findings

1. **Structured handoff is NOT better calibrated than the raw transcript.**
   ECE on active cases: handoff 0.248 vs transcript 0.239; truncation-headtail
   is the best-calibrated arm at ECE=0.207. The calibration benefit is null.

2. **All arms are severely overconfident.** Active-case mean confidence
   ~0.81–0.84; actual accuracy ~0.54–0.60. Gap ≈ 25 percentage points.
   This is a model-level calibration failure, not a representation problem.
   Context engineering alone cannot fix it.

3. **The confidence=0 abstention signal is uninformative.** Full-transcript
   abstains on 19 cases: 10 were actually correct answers, 9 were wrong —
   essentially a coin flip. The model does not know when it doesn't know.

4. **Risk-coverage does improve with threshold.** At confidence≥0.9, all arms
   improve to 67–82%, but coverage drops to 38–45%. A deployment requiring
   80%+ accuracy would need to reject more than half of cases — a constraint
   that must be reported alongside any accuracy number.

5. **handoff-plus-sources is the highest accuracy at threshold=0.9** (81.6%
   at 42% coverage), but the difference from full-transcript (75.6% at 45%
   coverage) is small and not tested for significance at this sub-n.

### Implications for model risk

These results quantify the gap between *self-reported confidence* and *actual
reliability* in a clinical diagnostic AI. A system reporting 82% average
confidence while being correct only 58% of the time cannot safely use its
own confidence scores for triage or escalation without external calibration
(temperature scaling, post-hoc Platt scaling, or conformal prediction).
None of the five context representations close this gap.

---

## Limitations

- 5O accuracy differences at n=30 are within overlapping CIs; the paired
  McNemar analysis lives in the 5O report and is not re-litigated here.
- Character-based token estimates (chars/4) are heuristic; exact per-condition
  input tokens come from provider metadata and are reported separately above.
- Negation/fabrication audits are lexical; the fabrication check also relies
  on the same fail-closed validator that produced the data (it is a
  consistency check, not an independent oracle).
- Stage 5O cases are now development evidence; nothing in this document was
  tuned on the future Phase 1 holdout, which had not been selected when this
  analysis ran.
