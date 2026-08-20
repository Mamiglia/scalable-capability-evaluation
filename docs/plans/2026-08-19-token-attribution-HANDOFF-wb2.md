# Token attribution — HANDOFF from wb2, 2026-08-19

Self-contained; written for a cold read. Supersedes `2026-08-18-token-attribution-HANDOFF.md`
(wb1's, still worth reading for the method-rejection history). The user-facing deliverable is
`judge_results/whitebox_dissection/` (report.md + figures + all-prompt dashboard + data/);
the full running log is `docs/results/2026-08-18-token-attribution-results.md`. Read both
before doing anything.

## One-line state

Llama-3.2-3B-Instruct is fully dissected and audited (attribution, validation, biases, two
internals stages); a 4096-draw fine-grain campaign is ~85% done on the cluster; the causal
stage, Qwen, and rubric-vs-verdict regressions await user decisions.

## What is established (all audited by fresh-eyes subagents; numbers in the report)

- Method: k = 8 random-token blanking (mean-embedding filler), sentence-level ridge with
  inner-CV penalty, per-prompt held-out ΔF gate (median R² 0.208, ceiling 0.78 → ~23–30% of
  explainable). k chosen by an additivity sweep; whole-span blanking provides the exact joint
  quantities (tenfold sub-additivity vs summed marginals).
- The judge favours the second-presented transcript (pair-summed verdicts −1.29 [−1.53,
  −1.07]); the deletion ledger reconstructs the lean (−0.76 vs −0.66); blank-prompt prior
  ≈ 0 (+0.012).
- Length bias: mechanism present (every sentence pushes its own slot; summed effects track
  length), effect capped by saturation (picks longer 54.3%; whole-span ΔF flat in length).
  Prediction: the stronger judge (Qwen) saturates less.
- The most influential sentences are conclusion declarations; transcript 2's "ANSWER: X"
  line pushes towards slot 2 in 76% of prompts (survives length matching, 3.6×) — the
  micro-carrier of recency, and the named target for the causal stage.
- Internals stage 1: L14 states predict sentence effects at held-out R² 0.183 vs 0.099
  identity+position (mid-network peak; robustness passes; global standardisation immaterial).
- Internals stage 2 (rank, user-approved role design): rank-1 beaten (0.124 → 0.233 at rank
  6), BUT audit shows the subspace concentrates on scaffolding + decision-adjacent transcript
  deciles; transcript-only rank-6 − rank-1 gap is 0.037. State it as scale-localised
  structure, not broad multi-directionality.
- Rubric: 15 items frozen, all 2,088 transcripts scored by haiku SUBAGENTS (never API keys),
  codex gpt-5.4-mini second-rated 50; seven items dropped by the user for failing
  cross-scorer validation (κ < ~0.3). Scores:
  `judge_results/whitebox_dissection/rubric_scores_haiku.parquet`. Human validation of the
  retained items is still required before any regression (frozen process).

## Running on the cluster right now

`maskcamp_k8_ext3584` (jobs 3393274-81, L40S batch 4): 3,584 extra draws/prompt, seed 1 —
pool with the 512-draw `maskcamp_k8_valid1106` for 4,096 total. ~88/111 parts done; ETA hours.
When it lands: pull both campaign dirs, run `judge-whitebox/mask_regression.py` at
`--level word` (and `token` if wanted; ~400 words vs ~1,700 tokens per prompt), regenerate
the dashboard with the word level (`scripts/whitebox_dissection_analysis/dashboard_v2.py`
auto-includes a level if `llama_word_scores.parquet` exists), and re-check the reliability
ceiling (should rise above 0.78).

## Pending user decisions

1. Causal stage go-ahead (project out fitted directions, measure verdict shift vs matched
   random directions; targets: the late-decile/answer-line directions). Protocol section
   "Testing the dimensions causally".
2. Qwen: everything deferred (user cancelled for GPU pressure). Recording already on ceph
   (`tokattr_valid1106/record_Qwen3-8B_all`); campaign/blank-groups/slot-pref resubmit with
   the same commands as Llama (swap WB_MODEL; Qwen chat template carries no date, immune to
   the date-pin issue).
3. Conclusion-statement framing in the report (user said they don't care about literal
   answer lines; the broader finding is conclusive-register sentences dominate).
4. Rubric human validation + then verdict regressions with retained items only.

## Where things live

- Repo (branch `analysis`, many commits, NOT pushed): `judge-whitebox/` — mask_regression.py
  (units: sentence/group/word/token), mask_summary.py, token_mediation.py (extraction,
  layer_fit, role rank machinery), token_attribution.py (mask/blank-groups/additivity modes),
  wbpairs (validity filter), wbprompts (TEMPLATE_DATE pin). 280+ tests green.
- Ceph `staging/whitebox/`: token_attribution/{maskcamp_k8_valid1106, maskcamp_k8_ext3584,
  blankgroups_valid1106, slotpref_full, addprobe_*}; token_mediation/feat_valid1106 (per-
  sentence features, re-pullable); token_manifold/tokattr_valid1106 (recordings, both judges).
- Rescued from the dead session scratchpad: analysis drivers in
  `scripts/whitebox_dissection_analysis/`, derived tables in
  `judge_results/whitebox_dissection/data/` (sentence/group scores, layer-curve CSV, rank
  outputs, codex validation scores). Campaign pulls and features: re-pull from ceph.
  NOTE: the driver scripts still reference the old scratchpad path — sed it to your session's
  scratchpad (or a stable dir) before running.

## Gotchas that cost time (do not rediscover)

- Selection identity: everything uses `--limit 1136 --limit-sampling balanced --limit-seed 0
  --min-reasoning-chars 100` (1,106 pairs). The old round-robin sampler concentrates
  DeepSeek-R1-Distill-70B into 63% of pairs — never use it for model-linked claims.
- Llama's chat template stamps the RENDER DATE into every prompt; wbprompts pins
  TEMPLATE_DATE = "18 Aug 2026". Never remove the pin; re-renders must match the corpus.
- L40S OOM: batch 32 × 9k-token prompts kills 44 GB; batch 4 is safe for the long tail,
  batch 8 for ≤5k. Slim submits (-c 4 --mem=32G, typed l40s gres) backfill in seconds where
  fat asks queue for hours. Campaign/blank-group modes RESUME from part dirs (per-prompt).
- cpu-partition nodes can read ceph at ~250 KB/s (25 s CPU in 13 min = the signature);
  extraction and anything I/O-heavy goes on GPU-partition nodes.
- Background processes: `cmd &` children die with the harness shell — use tracked
  run_in_background tasks with `wait`, and `< /dev/null` for codex (it blocks reading stdin).
- SPEND: model + channel + cost approved together, explicitly. OpenRouter is VETOED for
  scoring (user, hard). Scoring runs as haiku subagents (plan billing); codex CLI = user's
  OpenAI credit, cheap models only (`-m gpt-5.4-mini`, low effort).
- Sealed 40 items untouched; item-grouped CV and item bootstraps everywhere; nulls are
  results — never switch methods on the same data.
