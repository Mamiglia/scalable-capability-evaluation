# Quarterly ECI Frontier with Historical Judges

**Status:** Side analysis only. Nothing here feeds the canonical verdict-dissection dataset.

## Question

Can a judge infer a model's accuracy on unseen, harder items from pairwise comparisons of
successful transcripts on the easiest quartile? Difficulty is frozen once from the canonical
pool. Judges see Q1 only; predictions are evaluated separately on Q2, Q3, and Q4.

## Fixed frontier

The answer-model roster is the highest-ECI admissible model released in each quarter, fixed across
GPQA and SWE-bench mini. `low`, `high`, and `xhigh` answer variants are excluded; the ordinary
medium/default snapshot is retained.

| Quarter | Candidate | ECI |
|---|---|---:|
| 2024Q4 | GPT-4o 2024-11-20 | 129.06 |
| 2025Q1 | Claude 3.7 Sonnet | 141.63 |
| 2025Q2 | o3 | 147.02 |
| 2025Q3 | GPT-5 | 150.00 |
| 2025Q4 | Gemini 3 Pro Preview | 153.19 |
| 2026Q1 | Claude Opus 4.6 | 155.16 |
| 2026Q2 | Claude Opus 4.7 | 156.15 |
| 2026Q3 | Kimi K3 | 157.05 |

## Judges

The historical checkpoints are GPT-4o 2024-11-20, Gemini 2.5 Pro, and GPT-5.2. Existing o4-mini
and DeepSeek V4 Flash 0731 verdicts bridge the new candidates to the canonical pool. Every route is
single-provider, has fallbacks disabled, denies data collection, and records the provider actually
served on every response. GPT-5.2 alone disables OpenRouter's `require_parameters` registry check:
Inspect sends the valid `max_completion_tokens` alias, but OpenRouter's OpenAI endpoint does not
advertise that alias and otherwise rejects it before routing. The endpoint remains hard-pinned.

The pinned HF source is `LASR-G5/judging` revision
`6b2b9658f30de3aeade93d9192c32f4a0c19ce6e`. Only protocol-compatible GPQA and SWE-mini pilot
rows are reused. Invalid, superseded, reasoning-visible, truncated, non-counterbalanced, and
wrong-prompt rows are excluded.

## Protocol

- Difficulty comes only from the frozen canonical identity table; imported frontier outcomes never
  alter the quartiles.
- GPQA quartiles contain 49/50/49/50 items; SWE-mini quartiles contain 12/13/12/13.
- Judges see only Q1, and only items both answer models solved correctly.
- Q2, Q3, and Q4 are never judge inputs and are evaluated separately.
- Candidate reasoning is hidden (`include_reasoning: false`). GPQA uses
  `assets/system_prompt/mcq.txt`; SWE-mini uses `assets/system_prompt/default.txt`.
- Both presentation orders are required. There is no easy-half top-up and no data-dependent item
  expansion.

## Historical pair graph

Every paid target/opponent pair has a strictly older opponent. Claude 3.7 has GPT-4o; from o3
onward each target has the previous frontier model plus up to three diverse older pool anchors.
The exact 25 pairs live in `configs/frontier_eci/experiment.yaml` and are identical across
benchmarks.

Analysis is rolling. A candidate's BT fit contains only edges whose two answer models had been
released by that candidate's date. Later edges cannot revise earlier points. Absolute accuracy is
calibrated only on models strictly older than the target, using their known accuracy within the
same target quartile. Early candidates remain actual-only until at least four historical anchors
exist.

For each judge, report Q2/Q3/Q4 predicted accuracy, bootstrap interval, residual, MAE, Spearman,
position rate, reverse-order consistency, and longer-transcript preference. An equal-weight
consensus first averages repeated rounds within judge and then judges within exact
pair/item/order cells; upload volume cannot give one judge extra weight.

## Visibility diagnostic

The main protocol intentionally removes explicit reasoning blocks and inline `<think>` content.
The side diagnostic records, for correct Q1 transcripts, judge-visible characters, characters
added when reasoning is retained, reasoning-block prevalence, and redacted/encrypted reasoning
prevalence. It relates those observable features to prediction residuals. This is observational;
it is not presented as a matched CoT ablation. The pinned HF revision has reasoning-visible rounds
for AIME only, not GPQA or SWE-mini.

## Isolation

All material stays in:

- `configs/frontier_eci/`
- `scripts/frontier_eci/`
- `logs/frontier-eci/`
- `judge_results/frontier/eci-semiannual/`
- `data/<benchmark>/frontier-eci/`

The discarded preliminary frontier figures are provenance only. Canonical verdicts, difficulty,
freezes, and result directories are never modified.

## Execution and stopping

All downloads, log processing, API judging, analysis, and plotting run on the LASR RunPod. The
laptop is control-plane/file transfer only. Paid judging is resumable one pair at a time. Actual
OpenRouter key usage is checked before every pair against the experiment baseline, reserving that
pair's preflight worst-case cost. A small overage above $200 is allowed if needed to complete a
nearly finished run; exact API and RunPod spend are reported.

Final figures use the standing Nature style and are visually inspected at original resolution.
