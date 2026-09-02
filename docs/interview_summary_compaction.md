# Interview Narrative — Provenance-Aware Context Compaction for Interactive Clinical Agents

Every number below is real and reproducible from committed checkpoints in this
repo (`graphrag/eval/external/mediq/`, `graphrag/eval/data/`). Do not quote
anything you cannot point to a file for.

---

## The 2-minute story (English)

"I built an interactive clinical QA agent on the MediQ benchmark and turned it
into a controlled study of **context engineering**: what should an
information-gathering agent hand to a downstream reasoning model?

The agent interviews a simulated patient under a 3-question budget, retrieves
textbook evidence with BM25, and hands off to a **fresh diagnostic context**.
The core design is a provenance-linked structured handoff: every extracted
patient fact must cite the patient turn it came from, validated fail-closed —
a fact citing a nonexistent turn is rejected at parse time.

Three findings shaped the project.

First, a **negative result I kept**: adding reflection loops to a fixed
context increased cost without improving accuracy. That reframed the whole
project — agent value came from *information acquisition and context
management*, not from re-reasoning over the same context.

Second, on a frozen 30-case holdout with three matched arms, the structured
handoff scored **25/30 vs 22/30** for passing the raw transcript, and — the
part I care about most — **zero handoff-loss failures**: no case was right
from the transcript but wrong from the handoff. The audit showed 87% lexical
fact recall, 14/15 negated findings retained, and a **0/244 fabricated-fact
rate**, which validates the fail-closed provenance design.

Third, an **honest surprise**: at these short transcript lengths the handoff
is *not* a compression — it's about 2.1× the transcript size, +13% diagnostic
input tokens, because the JSON schema and provenance links cost more than a
3-question transcript. So its current value is auditability and noise
filtering, not token savings. That's exactly why my pre-registered follow-up
is a long-context stress test with injected distractor turns, where the
handoff size stays roughly fixed while the transcript grows.

I ran this like a validation exercise: frozen specs with seeds and dataset
fingerprints committed before execution, dev/holdout separation, paired
McNemar with reported ties, per-call cost guards, and every negative result
kept in the write-up."

## 30-second version

"I built an interactive medical QA agent and studied one context-engineering
question: transcript vs structured handoff into a fresh diagnostic context.
Provenance-linked handoff with fail-closed citation checking: 25/30 vs 22/30
on a frozen holdout, zero cases lost by compaction, zero fabricated facts in
244. The honest catch: at 3-question lengths it's not smaller than the
transcript, so the value is auditability — and I pre-registered a long-context
stress test to find where compression actually wins. Everything ran under
frozen specs, cost guards, and paired statistics."

## Problems I hit and what I tried (the "journey" answers)

1. **Reflection didn't help.** Early versions added reflect-and-retry loops.
   Ablations (Stages 5G–5J) showed extra cost, no accuracy gain. *Attempt →
   lesson:* stopped spending compute on re-reasoning; moved the intelligence
   into what context the diagnostician receives. I kept the negative result in
   the README as a governance finding.
2. **Context contamination.** Letting the diagnostician inherit the
   interviewer's full context made errors untraceable. *Attempt:* separated
   interviewer and diagnostician (AMIE-style), typed handoff packet, fresh
   context — every diagnostic answer now cites fact IDs that chain back to
   patient turns.
3. **Hallucinated facts.** An LLM summarizer can invent patient facts.
   *Attempt:* fail-closed schema validation — every fact must cite existing
   patient turn IDs, agent turns are rejected, uncited facts are rejected.
   Audit result: 0/244 fabricated facts on the holdout.
4. **The compression claim died on contact with data.** I assumed the handoff
   would be smaller. Measured on paid checkpoints (zero new calls): 2.1×
   *larger* at ≤3 questions. *Attempt:* reported it honestly, reframed the
   short-length value as auditability, designed a distractor-injection stress
   test to find the crossover point.
5. **Question selection is the real bottleneck.** Failure catalog: of 8 wrong
   cases, 5 were wrong in *all* arms — the discriminative fact was never asked
   for. No representation fixes that. *Next step:* a frozen pilot where the
   interviewer explicitly summarizes what the last question failed to yield
   before choosing the next one.

## Numbers you may be challenged on — exact honest phrasing

- **25/30 vs 22/30 (83% vs 73%)**: n=30, Wilson CIs overlap; paired discordant
  pairs were 3–0 in favor of the handoff, exact McNemar p=0.25. Say
  "directionally favorable, underpowered at n=30 — which is why the n=100
  pre-registered replication with five arms exists." Never say "significant."
- **0.874 fact recall / 0.607 precision**: lexical token-overlap heuristics,
  not semantic judgments. Low precision reflects extra context facts, not
  fabrications.
- **0/244 fabricated facts**: a consistency check of the fail-closed
  validator, not an independent oracle.
- **MediQ**: exam-derived cases, deterministic patient tool, uncalibrated LLM
  components — never claim clinical safety or deployment readiness.

## What NOT to claim

No clinical-safety claim, no compression claim at short lengths, no
statistical significance at n=30, no production deployment. If asked "did it
work?": "The accuracy signal is directional and the auditability result is
solid; the pre-registered n=100 run is designed to settle the primary
comparison."

## 中文教练笔记(不要背,理解逻辑)

- 这个故事最值钱的不是 83% vs 73%,而是三件事:**保留负结果**(reflection
  无效)、**诚实推翻自己的假设**(handoff 反而更大)、**验证纪律**(冻结
  spec、fingerprint、成本护栏、配对检验)。这三件事对 MRGR 面试官来说就是
  "model risk mindset" 的活案例,比任何漂亮数字都稀缺。
- 被追问统计功效时主动缴械:"n=30 配对不显著,我预注册了 n=100 五臂重复
  实验"——把弱点变成 next step,是最强的回答模式。
- 如果面试前跑完了 P1(n=100),把真实结果替换进 2 分钟版本;如果没跑完,
  就说 "the replication is specced, frozen, and costed at under $3" —— 这句
  本身就展示了成本意识。
