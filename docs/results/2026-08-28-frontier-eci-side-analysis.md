# Quarterly ECI frontier from Q1 judgments (side analysis)

**Status:** Run in progress. This is a non-canonical side analysis; it does not alter the frozen
canonical difficulty or verdict-dissection dataset.

## Design

The answer-model frontier is fixed across GPQA and SWE-mini using the highest admissible ECI model
per quarter, excluding `high`/`xhigh` variants. GPT-4o 2024-11-20, Gemini 2.5 Pro, and GPT-5.2 are
the historical checkpoint judges; o4-mini and DeepSeek V4 Flash 0731 provide bridge judgments to
the existing pool.

Difficulty quartiles are frozen exclusively from the canonical identity table. Judges see only Q1
items that both answer models solved. Q1 verdicts predict accuracy separately on unseen Q2, Q3,
and Q4 items. Every new target is compared only with strictly older opponents. Prediction fits are
also rolling: later comparisons cannot revise an earlier frontier point, and absolute calibration
uses only models strictly older than the target.

## Provenance

- HF judgments: `LASR-G5/judging` revision
  `6b2b9658f30de3aeade93d9192c32f4a0c19ce6e`.
- Frozen canonical identity SHA-256:
  `75f526757e8baec2c8b78e513f0930048aba88ffc4d3858c2f9bcfdf052b291b`.
- Frozen item-quartile SHA-256:
  `0db7cc8b38e6db8413a7b0fcca75d163ef61534e31bf6b91d2d1b17cf8000147`.
- Gemini 3 Pro SWE source SHA-256:
  `7f07c7ab8b1e6a6e553ecd62614662cbc1cee71a9e80439f2522a390cce92f9f`.
- Live endpoint snapshot: `judge_results/frontier/eci-semiannual/provenance/endpoints-2026-08-28.json`.
- Full configuration and pair graph: `configs/frontier_eci/experiment.yaml`.

The full run executes on the LASR RunPod only. Per-response served-provider and cost provenance is
exported after judging.

## Results

To be populated from the completed, verified tables and figures.

## Transcript visibility

The protocol strips explicit reasoning blocks and inline `<think>` content. A pre-result census
already shows a large public-transcript disparity: on correct SWE Q1 items, median visible
transcript length is about 7.9k characters for Opus 4.6 and 5.5k for Opus 4.7, versus roughly
35k–55k for the other frontier models. On GPQA, Opus 4.7's median visible answer is only 22
characters. Final residual analyses test whether this observable disparity explains the recurrent
Opus downrating. This is observational, not a matched CoT ablation; the pinned HF revision has no
reasoning-visible GPQA/SWE round.

## Outputs

- Predictions and metrics: `judge_results/frontier/eci-semiannual/analysis/`.
- Figures: `judge_results/frontier/eci-semiannual/analysis/figures/`.
- API provenance and cost: `judge_results/frontier/eci-semiannual/provenance/`.
- Paid-run ledger: `judge_results/frontier/eci-semiannual/run/spend-ledger.jsonl`.
