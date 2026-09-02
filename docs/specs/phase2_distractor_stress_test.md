# Phase 2 — Long-Context Distractor Stress Test (Pre-Registration)

**Status: FROZEN AND IMPLEMENTED — not executed. Requires owner approval before
any paid run.**

The executable spec, fixed distractors, and runner were committed in
`4e707dc`. The current dry-run estimates 300 new calls and about $0.30; the
hard runtime guard remains $2.50 to fail safely if token usage is higher than
the simple per-call estimate.

## Motivation

Phase 0 and Phase 1 both find no significant accuracy advantage for the
structured handoff at ≤3 interview questions. The compression-accounting
analysis shows handoff costs +20% diagnostic tokens over raw transcript at
current lengths. The theoretical compression claim requires longer, noisier
transcripts — where the handoff size is roughly fixed while the transcript
grows. Phase 2 tests this directly.

Phase 0 also noted: the handoff has a **tighter token ceiling** (Phase 0
max 2,945 tokens vs transcript max 3,844). Phase 2 probes whether that
ceiling matters under distractor injection.

## Design

### Dataset

- Source: the Phase 1 holdout (`graphrag/eval/specs/mediq_compaction_p1.json`,
  seed 47, fingerprint `9e3e9df2925823ed`).
- Subset: 30 cases selected deterministically (seed 2, stratified by specialty
  6 per specialty) from the 91 Phase 1 evaluated cases. No additional model
  calls for selection — indices derived from existing results.
- All 35 Stage 5N/5O IDs and the 30-case Phase 2 subset IDs must be committed
  in a frozen JSON spec before any run.

### Distractor injection

- **k ∈ {0, 10, 25} distractor turns** injected into each transcript.
- Distractor turns are generated ONCE from a separate deterministic seed (seed
  42), committed as a JSON artifact alongside the spec, and REUSED identically
  across all arms. No per-run generation.
- A distractor turn = a plausible-but-diagnostically-irrelevant Q&A pair drawn
  from a held-aside MediQ pool (disjoint from all study cases). Format matches
  existing interview turns.
- The handoff is rebuilt from the padded transcript (including distractor
  turns) so the handoff also grows with k — this tests whether the handoff
  extracts signal from noise better than the raw transcript.

### Arms

| Arm | Description |
|---|---|
| full-transcript | Raw transcript with k injected distractors |
| freetext-summary | One `summarize` call on the padded transcript |
| structured-handoff | Handoff rebuilt from padded transcript |

Truncation-headtail and handoff-plus-sources are dropped from Phase 2 to
contain cost. They can be added in a later run if Phase 2 finds a signal.

### Primary comparison

Structured-handoff vs full-transcript at k=25 (the hardest condition).
McNemar paired test, n=30 per k level.

### Secondary comparisons (exploratory, labelled as such)

- Accuracy-vs-k curve for each arm (k=0, 10, 25) — is there a degradation
  crossover point?
- Token ratio (handoff / transcript) at each k — does the handoff actually
  compress as context grows?
- freetext-summary vs full-transcript at k=25 (does lossy summarization beat
  raw transcript under noise?)

### Pre-registration rules

Per the repository's non-negotiable experiment rules:

1. The distractor JSON must be committed and fingerprinted before any model
   call.
2. No arm selection, threshold, or metric changes after seeing results.
3. Primary comparison is k=25 structured-handoff vs full-transcript.
4. Report all three k levels; no selective reporting.
5. If transcript doesn't degrade at k=25, report that honestly.

## Budget

- 30 cases × 3 k-levels × 3 arms × (interview replay-free + 1 diagnose) ≈ 270
  diagnose calls + 30 freetext-summary calls.
- Interview turns already checkpointed (Phase 1 checkpoint reused for the
  30-case subset; distractors appended in the runner, no new interview calls).
- Current runner estimate: ~$0.30. Hard guard: $2.50.

## Decision gate

**STOP here.** Before executing Phase 2, the owner must weigh this against
Phase IG (`docs/specs/information_gain_interviewing.md`, ~$1.2 pilot + ~$2.3
holdout). Both phases compete for the remaining budget (~$8.67 after Phase 1).
Raise the trade-off explicitly:

- Phase 2 answers: does representation quality matter when context is long?
- Phase IG answers: does question selection matter more than representation?
- Phase 0 identified question selection as the primary bottleneck (5/8 failures
  = information never acquired). Phase 1 confirms no representation wins at
  short transcripts. The bottleneck argument favours Phase IG first.

Owner decides.
