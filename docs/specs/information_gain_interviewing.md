# Pre-Registration — Information-Gain-Guided Interviewing (Phase IG)

**Status:** frozen design, NOT executed. Requires owner approval before any
paid run. Written before seeing any data from this experiment.

**Motivation (from committed evidence, not speculation):** the Phase 0 failure
catalog (`docs/compression_analysis.md`) showed that 5 of 8 Stage 5O failures
were wrong in *every* diagnostic arm — the discriminative fact was never
elicited within the 3-question budget. No downstream representation can
recover information that was never acquired. Every arm in Phase 1 varies how
the transcript is *represented*; none varies how questions are *selected*.
This phase targets the bottleneck the data actually identified.

---

## 1. Research question and hypotheses

**RQ:** Does choosing interview questions by expected information gain over
the answer distribution beat letting the LLM choose questions freely, under a
fixed question budget?

- **H1 (headroom).** An oracle-assisted information-gain selector outperforms
  free-form LLM question selection on final answer accuracy.
  *Pre-registered primary comparison:* `oracle-ig` vs `llm-freeform`.
- **H2 (deployability).** A deployable selector that must *imagine* patient
  answers captures a meaningful share of the oracle headroom.
  *Secondary:* `simulated-ig` vs `llm-freeform`, and the ratio
  (simulated − freeform) / (oracle − freeform).
- **H3 (efficiency, exploratory).** Information-gain selection reaches its
  final answer in fewer questions, i.e. accuracy at question budget 1 and 2 is
  higher even when budget-3 accuracy is similar.

**Falsification is a real outcome.** If `oracle-ig` does not beat
`llm-freeform`, the honest conclusion is that the LLM's free-form question
choice is already near the achievable ceiling under this patient simulator,
and that the residual errors are intrinsic to the case or the diagnostic step
rather than to question selection. That result gets reported, not buried.

## 2. Method

At interview turn *t* with current handoff `H_t`:

1. **Candidate generation.** One model call produces `M = 3` distinct
   candidate questions (instead of today's single question). Same interviewer
   contract otherwise.
2. **Answer acquisition per candidate** — this is where the arms differ:
   - `oracle-ig`: query the *deterministic* patient tool
     (`ConceptAwareFactPatient`) with each candidate. **Zero model cost** —
     the simulator is a local deterministic fact matcher.
   - `simulated-ig`: the model imagines the likely patient answer for each
     candidate, without touching the patient tool. Produced in the same call
     as candidate generation.
3. **Scoring.** For each candidate `q_m` with (real or imagined) answer `a_m`,
   one diagnostic call returns a probability distribution over the case's
   answer options given `H_t + (q_m, a_m)`.
4. **Selection.** Choose the candidate maximising entropy reduction

   ```
   ΔH(q_m) = H(P(answer | H_t)) − H(P(answer | H_t + (q_m, a_m)))
   H(P) = − Σ_i p_i log p_i        (natural log, options only, renormalised)
   ```

   Ties broken by lowest candidate index (deterministic).
5. **Commit.** Only the selected question is actually asked; the handoff is
   updated by the existing interviewer path. Discarded candidates leave no
   trace in the handoff.

Baseline `H(P(answer | H_t))` is computed once per turn (one extra call),
reused across candidates.

### Why "expected" is not what we compute

The patient simulator is deterministic, so for `oracle-ig` the patient's
answer to a candidate is a single known value: we compute **realized**
information gain, not an expectation over an answer distribution. In a real
clinical setting the answer is stochastic and the correct quantity is
`E_a[ΔH(q, a)]`. This is a stated limitation, not an oversight, and it is the
main reason `oracle-ig` is labelled an idealised upper bound.

### Why `oracle-ig` is not deployable

It probes the patient with all `M` candidates before committing to one. A real
interview cannot un-ask a question. `oracle-ig` therefore measures *headroom*
— how much accuracy is reachable if question selection were solved — and
`simulated-ig` measures how much of that headroom a deployable method reaches.
This distinction must survive into the README and any interview answer.

## 3. Arms (all share the same diagnostic representation)

Only the question-selection policy varies. The diagnostic input is
`structured-handoff` in every arm, so this experiment is orthogonal to Phase 1.

| Arm | Question selection | Patient tool used for scoring | Deployable |
|---|---|---|---|
| `llm-freeform` | current interviewer (baseline) | no | yes |
| `oracle-ig` | max ΔH, real answers | yes | no (upper bound) |
| `simulated-ig` | max ΔH, imagined answers | no | yes |

Matched across arms: same model (`gemini-2.5-flash`), same 3-question budget,
same retriever (BM25 top-k 8), same handoff schema, same diagnostic prompt,
same patient simulator, same coverage tool.

## 4. Data

New selections; all previously used source IDs excluded (5N=5, 5O=30,
P1=100 → 135 excluded).

- **Development set:** n=20, seed 71, 4 per specialty. Prompt iteration,
  candidate count, and any debugging happen **here only**.
- **Holdout:** n=40, seed 83, 8 per specialty, disjoint from development and
  from all prior sets. Frozen before execution; run **once**.

Both selections must be committed as a JSON spec with a fingerprint, in the
same format as `graphrag/eval/specs/mediq_compaction_p1.json`, before any
paid call. Selection is regenerated at load time and must match exactly.

## 5. Statistics

- Primary: paired exact McNemar, `oracle-ig` vs `llm-freeform`, on the 40-case
  holdout. Wilson 95% CIs per arm. Ties reported.
- **Power is the binding constraint, and n=40 is small.** With ~10% discordant
  pairs (the Stage 5O rate) only ~4 discordant pairs are expected, which
  cannot support a significance claim. This phase is therefore explicitly
  **estimation-oriented**: report the effect size with its interval and treat
  it as a signal for whether a larger run is worth funding. Do not report a
  p-value as if it settled anything.
- H3 uses accuracy at truncated budgets (1, 2, 3 questions), reconstructed
  from the same run — no extra calls.

## 6. Cost

Per case, approximate logical calls:

| Arm | Calls/case | Note |
|---|---|---|
| `llm-freeform` | ~5 | 4 interview + 1 diagnose |
| `oracle-ig` | ~16 | 3 turns × (1 candidate-gen + 1 baseline + 3 scoring) + 1 diagnose |
| `simulated-ig` | ~16 | same, imagined answers ride in the candidate-gen call |

- Development pilot (n=20, all arms): ~740 calls ≈ **$1.2**, guard **$1.8**
- Holdout (n=40, all arms): ~1,480 calls ≈ **$2.3**, guard **$3.0**

Run the pilot first, stop, report, and get approval again before the holdout.
Both stay inside the remaining project budget only if Phase 2 is descoped or
deferred — **flag this trade-off to the owner explicitly before spending**.

## 7. Honesty rules carried over

1. Freeze and commit the spec + selections before execution.
2. No tuning on holdout after seeing results; deviations need a new spec.
3. Report `oracle-ig` as an idealised upper bound everywhere, never as a
   deployable result.
4. Keep negative results; a null H1 is a publishable finding for this repo.
5. Per-call cost guard; stop for approval at each paid stage.

## 8. Known limitations to state up front

- Deterministic patient simulator → realized, not expected, information gain.
- MediQ is exam-derived; answer options exist, which is what makes the entropy
  computation possible at all. **This method does not transfer unchanged to
  open-ended diagnosis**, where there is no finite option set — a natural
  follow-up is entropy over a sampled differential.
- Probability outputs from an LLM are not calibrated. Entropy computed from
  uncalibrated probabilities ranks candidates plausibly but is not a
  well-founded uncertainty measure. Calibration (e.g. reliability curves on
  the development set) is a stated gap.
- Single model, single dataset; no cross-model generalisation claim.
