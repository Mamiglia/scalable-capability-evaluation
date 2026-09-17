# Closed frontier models as coding targets (2026-I)

Do the judging protocols still forecast θ* when the targets are closed frontier models? Coding only,
2026-I task only. No collection or judging is launched from this document; the trial run and the full
runs each need an explicit go.

## Roles

**Targets (2026-I).** The 8 open targets, unchanged and already judged, plus 6 closed models:

| Model | Released | New collection |
| --- | --- | --- |
| Claude Fable 5 | 2026-06 | train + HARD subset |
| Claude Opus 4.8 | 2026-05 | train + HARD subset |
| GPT-5.5, reasoning effort xhigh (not Pro) | 2026-04 | train + HARD subset |
| Gemini 3.1 Pro | 2026-02 | train + HARD subset |
| Claude Opus 4.7, thinking off | 2026-04 | AgentBench OS only (SWE + Terminal-Bench already collected) |
| Gemini 3.5 Flash | 2026-05 | AgentBench OS only (SWE + Terminal-Bench already collected) |

GPT-5.5 Pro is excluded: OpenAI gives Pro models no cached-input price at any effort, so it cannot
be collected with caching (~$2.3k for one model).

**References (2025-II).** Unchanged: kimi-k2-thinking, minimax-m2, deepseek-v3.2, glm-4.5, glm-4.7,
qwen3-235b-a22b-thinking-2507, gpt-oss-120b, gpt-oss-20b. The historical cohort for the rating
judge's calibration is unchanged.

**Judges.**

- Kimi-K2-Thinking, the best open coding 2026-I judge (pairwise linear MSE 0.11). Judges the 6
  closed targets only; the open targets' verdicts exist.
  - pairwise, reasoning shown, all 100 train items (existing condition `24d6cc32`)
  - pairwise, reasoning hidden, 40-item subset (existing condition `9c1d3abb`; no full-set hidden run exists)
  - rating judge, existing condition
- GPT-5.2 (released 2025-12), the one closed 2025-II judge. Judges everything from scratch on the
  40-item subset (10 per difficulty quartile of the frozen train set), reasoning shown:
  reference-vs-reference pairs (they calibrate the linear conversion), the 8 open targets and the 6
  closed targets; plus its rating judge over references, historical cohort and all 14 targets.

Closed models never return raw chain-of-thought, so under "reasoning shown" the judge sees the
references' reasoning and not the closed targets'. The reasoning-hidden Kimi condition measures what
that asymmetry costs.

## Items

The frozen `2026-I_train` column of `data/meta/coding/items.csv`, never widened or redrawn: 100 items
(AgentBench OS 65, Terminal-Bench 2.1 22, SWE-bench mini 13).

θ* needs HARD-item responses. The 4 newly collected models run the 40 HARD items with the most 2PL
information averaged over θ 1.5-3 (Terminal-Bench 23, SWE 15, AgentBench 2): θ* standard error 0.18
at θ 2.5, against 0.16 with all 107. The list is computed from frozen item parameters only and written
to the campaign's frozen inputs before collection. Opus 4.7 and Gemini 3.5 Flash use all their existing
SWE and Terminal-Bench HARD responses plus the 2 AgentBench items.

## Collection

Same scaffolds and limits as the open roster, so transcripts differ only by model: SWE-bench mini
react(bash + python) with 100 messages; Terminal-Bench 2.1 with 200 messages; AgentBench OS task args
unchanged (hash `cc9e5147`); per-turn `max_tokens` 32768. Providers pinned first-party (Anthropic,
OpenAI, Google Vertex / AI Studio). Reasoning: Fable 5 and Opus 4.8 with thinking enabled at default
effort, GPT-5.5 at xhigh, Gemini 3.1 Pro at its default thinking level.

Epoch AI's logs are not used for judged transcripts: they cover only SWE-bench Verified (46 of our 50
items, at most 13 train items), only Gemini 3.1 Pro / Opus 4.7 / GPT-5.5 of this roster, and use a
different scaffold (text editor + apply-patch tools, 2M-token budget), which would enter every
pairwise comparison.

The logs select items with `--sample-id`, which `data/CONTRACT.md` does not accept as a complete run,
so they live in `artifacts/closed-frontier-coding/` (harness in `src/challenges/closed_frontier/`) and
are published to the judging repo like the truncation campaign. Frozen `data/meta/coding/` tables are
not refit.

## Caching

- Collection, Anthropic: inspect 0.3.263's OpenRouter provider adds `cache_control` breakpoints by
  default. The August Opus 4.7 Terminal-Bench run got 8% cache reads, so this is verified, not assumed.
- Collection, OpenAI and Gemini: automatic prefix caching (80-88% cache reads on our GPT-5 and Gemini
  3.5 Flash logs).
- Judging: prompts start with the system prompt and the first transcript, which repeat across pairs;
  GPT-5.2 caches these automatically. Hit rate is reported, not relied on in the estimate.
- Inspect's response cache on for every run, so a resumed run does not pay again for calls it already made.

## Safeguards

1. A dedicated OpenRouter key with a hard credit limit equal to the approved budget. The current key is
   at its weekly limit.
2. Trial run: 2 samples per benchmark per newly collected model, plus 20 GPT-5.2 verdicts (~$30).
   Pass = cache reads at least 70% of input tokens on SWE and Terminal-Bench samples, and measured cost
   per sample within 1.5x of the estimate. Fail on Anthropic caching = switch those models to the
   Anthropic API directly (needs a key). The measured costs replace the estimates below before the full go.
3. Full runs cheapest first: AgentBench OS, then SWE, then Terminal-Bench; `--fail-on-error 0.1`.
4. Everything runs on loki; nothing is stored on the laptop.

## Estimated cost

| Item | Estimate |
| --- | ---: |
| Fable 5 collection | ~$210 |
| Opus 4.8 collection | ~$105 |
| GPT-5.5 xhigh collection | ~$145 |
| Gemini 3.1 Pro collection | ~$75 |
| Opus 4.7 + Gemini 3.5 Flash, AgentBench OS | ~$10 |
| Kimi judging, 6 closed targets, both conditions + rating | ~$135 |
| GPT-5.2 judging, 40-item subset, pairwise + rating | ~$450 |
| Trial run | ~$30 |
| **Total** | **~$1,160** |

Collection estimates assume caching works; without it Anthropic models cost ~3x (Fable 5 ~$620,
Opus 4.8 ~$310). Judge estimates use the token counts of Kimi's 518 coding 2026-I logs (163M input
tokens for the open cohort), before caching.

## Analysis

θ* for each closed model: fixed-item 2PL (`irt.fitting.fixed_2pl`) with the frozen a, b on its HARD
responses. Forecasts per judge and condition with the existing conversions (linear, calibrated,
Bradley-Terry for pairwise; raw and item z-scores for rating). MSE against θ* reported for the open
targets, the closed targets and all 14, next to easy-item prediction. θ* standard errors are reported
per closed model, since theirs rest on 40 HARD items.
