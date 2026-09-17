# Transcript-feature regression — handoff

Status 2026-09-15 evening on `methodology`: all three tiers implemented, extracted, fitted and pushed; see the status section below.

## Agreed method

Predict each target model's domain-specific HARD-item theta from deterministic features of its correct, first-attempt transcripts on the selected EASY items. Fit on the labelled union of **all earlier semesters** available for that task; this intentionally differs from pairwise judging, which uses only the immediately previous semester.

The three nested ridge regressions are:

| Protocol | Predictors |
| --- | --- |
| `glm_length` | 3 length/reasoning-visibility features |
| `glm_regex` | the above plus 9 lexical rates (12 total) |
| `glm_structure` | the above plus tool count, tool action share, 6 selected transitions and 4 source-observed flags (24 total) |

Rates are calculated per transcript and then averaged by model. Feature extraction must not call a generative LLM. IsaNLP RST parsing is allowed. Tool calls and outputs remain structured nodes; calls should be split into inspection/information-gathering versus action. Exact definitions and exclusions are in [the feature table](../transcript-feature-table.md).

## Status 2026-09-15 evening

All three tiers run end to end from the per-transcript feature CSVs in `glm_features/`
(clone of `LASR-G5/glm_features`, gitignored; same layout as `data/`). Fitting needs no logs, so
`data/` stays as LFS pointers (25 GB otherwise; the laptop cannot hold it).

- `scripts/glm_features/extract.py`: one CSV per successful log, one row per sample and
  reasoning view (`visible` / `hidden`), the 13 length/regex/tool-count columns. 4,000 CSVs.
- `scripts/glm_features/export_transcripts.py` + `rst_labels.py` + `merge_labels.py`: the
  structure tier. Every correct first attempt on each task's items (14,957 transcripts) was
  parsed with `tchewik/isanlp_rst_v3` (gumrrg, revision eb1d5745) on Runpod GPUs (two pods,
  ~2.5 h, ~$4); labels live beside each CSV as `<eval id>.labels.jsonl` and the 11 structure
  columns are filled for both views. 21 transcripts have one very long reasoning part
  (95k to 300k chars) the parser rejected; their labels omit that part.
- `TranscriptRegression`: `StandardScaler` + `RidgeCV(logspace(-3, 3, 13))`, leave-one-out on
  the training cohort. `glm_length`, `glm_regex`, `glm_structure` are in `PROTOCOLS`;
  `uv run evaluate configs/domain/<d>.yml --protocol glm_length glm_regex glm_structure`.
- `scripts/plots/plot_glm_forecasts.py`: true vs predicted theta* per domain and semester.

MSE (theta units), science / math / coding / cyber:

| semester | length | regex | structure |
| --- | --- | --- | --- |
| 2024-I | 1.65 / 2.25 | 1.66 / 1.97 | 1.68 / 1.17 |
| 2024-II | 0.54 / 0.36 | 0.58 / 0.42 | 0.55 / 0.39 |
| 2025-I | 0.97 / 0.91 | 38.1 / 578 | 6.9 / 395 |
| 2025-II | 0.65 / 1.21 / 2.64 / 4.00 | 0.28 / 0.89 / 0.39 / 0.89 | 0.35 / 0.91 / 0.28 / 1.02 |
| 2026-I | 0.40 / 0.30 / 2.18 / 1.86 | 0.57 / 0.63 / 0.90 / 1.84 | 0.84 / 0.38 / 1.17 / 2.04 |

Known limits:

- **2025-I explodes for regex and structure.** The frozen tables carry no 2025-I theta* for the
  2024-I models, so the training cohort is the 8 previous-semester references, none of which
  shows reasoning; the targets are the first reasoning models and sit 50 to 250 training
  standard deviations out on `backtracking`, `questions`, `hedging`. Standardised ridge then
  extrapolates linearly. Fixing this needs either historical theta* for those models or a
  bounded feature transform; it is not a counts-vs-rates issue.
- 2024-I predictions are flat: 8 references and no history, so leave-one-out picks the
  maximum penalty.
- The `data/` submodule holds 30 crashed/partial logs beside successful runs (18 repeat dirs);
  the extractor skips them. `intercode/463eb861` also has 9 sample-less stubs.
- The frozen science table points at a gpqa log for `gemma-4-26b-a4b-it` that no longer exists
  at the data pin (the model has no role in any semester, so nothing depends on it).
