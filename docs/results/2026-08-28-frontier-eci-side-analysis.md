# Quarterly ECI frontier from Q1 judgments (side analysis)

**Status:** GPQA complete; SWE-mini partial because the Gemini 3 Pro source archive was unavailable
locally. This is a non-canonical side analysis and does not alter the frozen canonical difficulty
or verdict-dissection dataset.

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
  `6b2b9658f30de3aeade93d9192c32f4a0c19ce6e`: all 1,493 GPQA/SWE pilot files,
  yielding exactly 110,748 compatible verdict rows. The T7 mirror lacked 296 pinned files
  (671 MB) and contained 99 older extras; missing files were downloaded into the isolated side
  directory and an exact pinned-path allowlist excluded the extras.
- Frozen canonical identity SHA-256:
  `75f526757e8baec2c8b78e513f0930048aba88ffc4d3858c2f9bcfdf052b291b`.
- Frozen item-quartile SHA-256:
  `0db7cc8b38e6db8413a7b0fcca75d163ef61534e31bf6b91d2d1b17cf8000147`.
- Full configuration and pair graph: `configs/frontier_eci/experiment.yaml`.

The first RunPod attempt stopped when its OpenRouter account exhausted its balance. Its completed
rows are not inputs here: the complete matrix was rerun locally with the personal key. Provider
audit confirms every response came from the intended endpoint (OpenAI, Google AI Studio, or
DeepInfra). The local matrix contains 11,562 API responses and cost $109.39. Including smoke tests
and the abandoned RunPod calls, total experiment API spend was about $130.6.

## GPQA results

All judges systematically underpredict unseen harder-quartile accuracy. Across the six frontier
models with enough historical anchors, checkpoint-judge MAE on Q2/Q3/Q4 is:

| Judge | Q2 | Q3 | Q4 |
|---|---:|---:|---:|
| GPT-4o 2024-11-20 | 0.263 | 0.445 | 0.293 |
| Gemini 2.5 Pro | 0.097 | 0.322 | 0.230 |
| GPT-5.2 | **0.079** | **0.249** | **0.218** |

Every mean residual is negative. The equal-judge consensus has MAE 0.125/0.379/0.246 and does not
beat GPT-5.2. Ranking is more successful than magnitude on Q3/Q4: Gemini and GPT-5.2 reach
Spearman 0.94 on Q3, while every judge gets the Q2 ordering wrong (Spearman −0.56 to −0.68).

Consensus mean error by answer model is: GPT-5 −0.317, Kimi K3 −0.287, Opus 4.7 −0.280, o3
−0.265, Gemini 3 Pro −0.256, and Opus 4.6 −0.094. Thus the corrected experiment does **not** find
uniform Opus downrating: Opus 4.7 is strongly underrated, but Opus 4.6 is the best-calibrated
frontier model on GPQA. GPT-5 is worst.

Counterbalancing is present in the inputs but does not remove judge behavior: first-position rates
range from 0.37 (o4-mini) to 0.48 (GPT-4o), and judges pick the longer transcript 61–84% of the
time. GPT-4o has the strongest length preference (0.835).

## SWE-mini partial results

All 93 non-Gemini configured shards completed. Gemini 3 Pro itself and six pairs that require its
missing 4 GB source archive remain explicit gaps; the eight-model quarterly roster is not silently
changed. Only four later candidates have enough anchors for scored forecasts, so these plots are
diagnostic rather than the final cross-benchmark result.

The same underprediction appears: checkpoint MAE on Q2/Q3/Q4 is 0.302/0.370/0.199 for GPT-4o,
0.343/0.396/0.203 for Gemini 2.5 Pro, and **0.109/0.327/0.168** for GPT-5.2. Consensus mean error is
−0.282 for Kimi K3, −0.260 for Opus 4.7, −0.258 for Opus 4.6, and −0.173 for GPT-5.

## Transcript visibility

The protocol strips explicit reasoning blocks and inline `<think>` content. On GPQA, mean residual
has Spearman 0.77 with visible transcript length: models exposing more text are generally less
underpredicted. Opus 4.7 exposes a median 22 characters and is underrated by 0.280; GPT-5 exposes
421 characters and is underrated by 0.317. o3, GPT-5, Gemini 3 Pro, and Opus 4.6 have redacted or
encrypted reasoning on essentially every correct Q1 sample, so the judge is calibrating from a
surface that systematically omits their private computation.

This is not the whole explanation. Opus 4.6 is only 0.094 low on GPQA despite redacted reasoning,
and Kimi K3 is 0.287 low despite a longer visible answer. On partial SWE, Opus 4.6/4.7 are by far
the shortest visible transcripts (7.9k/5.5k versus 35k–55k for the others) and both are about 0.26
low, but length–error Spearman over the four forecastable models is −0.20. The evidence supports a
visibility/style channel, not a claim that stripping CoT alone causes the error.

Encrypted CoT cannot be recovered fairly for selected models. A defensible follow-up is a separate
matched public-rationale protocol—ask every model for the same short, visible post-hoc explanation
and calibrate that surface—rather than showing private CoT only for models that expose it. The
pinned HF revision contains no reasoning-visible GPQA/SWE round, so no causal CoT ablation is
reported here.

## Outputs

- Complete GPQA predictions, metrics, and figures:
  `judge_results/frontier/eci-semiannual/analysis/`.
- Partial SWE predictions, metrics, and figures:
  `judge_results/frontier/eci-semiannual/analysis-swe-partial/`.
- API provenance and cost: `judge_results/frontier/eci-semiannual/provenance/`.
- Paid-run ledger and 20-minute monitors:
  `judge_results/frontier/eci-semiannual/run-local/`.
