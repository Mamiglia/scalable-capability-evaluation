# Task 5 report: canonical forecasts, paired ECI baseline, and release-time figures

## Outcome

Task 5 completed. The final analysis contains all five judges, both benchmarks, Q2/Q3/Q4, capped headline forecasts, uncapped sensitivity, an exact mapped-anchor ECI diagnostic, and a judge sensitivity refit on the identical ECI-available anchor subset. The headline judge anchor sets remain unchanged by ECI availability.

The final frozen coverage is 79/80 identifiable. The remaining explicit gap is SWE-mini / GPT-4o judge / GPT-4o target, with only two attainable anchors; it is not imputed. All 240 prediction rows have target hard outcomes, including the recovered Gemini 3 Pro SWE outcomes.

## Files

Code and tests:

- `scripts/frontier_eci/canonical_analysis.py`
- `scripts/frontier_eci/canonical_capability_baseline.py`
- `scripts/frontier_eci/analyze.py`
- `tests/frontier_eci/test_canonical_analysis.py`
- `tests/frontier_eci/test_canonical_capability_baseline.py`
- `tests/frontier_eci/test_analysis.py`
- `configs/frontier_eci_canonical_anchors/experiment.yaml`

Judge outputs under `judge_results/frontier/eci-semiannual-canonical-anchors/analysis/`:

- `predictions.csv`, `predictions_uncapped.csv`
- `anchor_fits.csv`, `anchor_fits_uncapped.csv`, `anchor_residuals.csv`
- `metrics.csv`, `metrics_uncapped.csv`, `model_residuals.csv`
- `coverage.csv`, `summary.json`
- `figures/frontier_style/`: 60 plot stems (30 judge/benchmark/quartile panels × band/points), each PNG/PDF/SVG = 180 files

Matched baseline outputs under `judge_results/frontier/eci-semiannual-canonical-anchors/capability_baseline/`:

- ECI `predictions.csv`, `anchor_fits.csv`, `anchor_residuals.csv`, `metrics.csv`, `model_residuals.csv`, `coverage.csv`
- `eci_model_mapping.csv` with every reviewed internal-ID → Epoch `Model` identity or explicit absence
- `paired_judge_predictions.csv`, `paired_judge_anchor_fits.csv`, `paired_judge_anchor_residuals.csv`, `paired_judge_metrics.csv`
- `summary.json`
- `figures/`: 60 ECI plot stems, each PNG/PDF/SVG = 180 files

There are 190 analysis files (5.4 MB) and 192 baseline files (5.5 MB). Metric tables have 87 capped rows, 87 uncapped rows, 75 ECI rows, and 90 paired-judge rows; absent inside/outside groups are omitted rather than fabricated.

## RED/GREEN evidence

Initial matched-ECI RED:

```text
$ uv run pytest tests/frontier_eci/test_canonical_capability_baseline.py -q
FileNotFoundError: scripts/frontier_eci/canonical_capability_baseline.py
```

The first GREEN attempt exposed a real ECI-scale conditioning failure: raw predictors around 100–150 made SciPy report optimizer success while the original-scale weighted gradients were approximately 0.15 and 0.038. A dedicated large-offset regression failed before the fix. The first conditional-centering fix passed locally but the RunPod BLAS path produced architecture-sensitive bootstrap success counts and precision-loss failures. The final deterministic two-parameter Newton/IRLS implementation uses the analytic gradient/Hessian with backtracking, transforms the explicit intercept/slope back to raw units, and accepts only original-scale weighted gradients below `1e-6`.

Additional RED cycles covered:

- frozen Q1 + exact-epoch both-correct filtering;
- reviewed `gemma3-12b-it` transcript-key normalization to canonical `gemma-3-12b-it`;
- all connected non-frontier canonical anchors and backfill only below four;
- uncapped sensitivity, range-stratified metrics, explicit frozen failures, and actual-only supplement precedence;
- narrow PNG/PDF/editable-SVG figure output;
- ECI hard-item bootstrap bands, exact one-row alias validation, mapped subsets, dropped-ID export, and headline-anchor immutability;
- spawn-based block multiprocessing equivalence with stable SHA-256 block seeds.

Final focused GREEN:

```text
$ uv run pytest tests/frontier_eci/test_canonical_analysis.py \
    tests/frontier_eci/test_canonical_capability_baseline.py \
    tests/frontier_eci/test_analysis.py \
    tests/frontier_eci/test_capability_baseline.py -q
36 passed
```

Final full GREEN:

```text
$ uv run pytest -q
190 passed, 1 pre-existing Pydantic deprecation warning in 8.20s
```

Python compilation, both RunPod launcher `bash -n` checks, and `git diff --check` also exited zero.

## Gemini 3 SWE completion and targeted paid top-up

The initial full-data smoke made the five SWE Gemini 3 blocks visibly missing and left 15 actual cells absent (five judges × three quartiles). The exact public Epoch artifact was therefore recovered without imputation:

- artifact ID: `5XdYZBG5Wy8DUMRqcYv4fn`
- raw S3 size: 3,999,137,978 bytes
- trimmed with the existing SWE importer to the frozen item IDs and 100-message cap
- source provided 46/50 frozen IDs; four were absent upstream
- compact size: 3,594,936 bytes
- compact SHA-256: `16dfd92eae2b8ca3802b3ace4396ccafa58eb0f09b30cfcc3dd24b28822129fd`
- available Q1/Q2/Q3/Q4 counts: 12/13/12/9
- available accuracies: 1.000 / 0.846 / 0.417 / 0.111

The 46 outcomes were atomically added to the ignored side `new-tables/rollouts.parquet` (5,044 → 5,090 rows, zero duplicate benchmark/model/item/epoch keys). The portable `actual_supplements` entry remains as a reproducible fallback; normal analysis precedence uses the side table and does not duplicate it.

Coverage regeneration then produced 79 identifiable blocks, one unavoidable missing block, 1,713 pair-item rows, and 245 configs. Exactly 30 configs lacked successful logs; all were newly enabled SWE Gemini-3 target pairs, six per judge. All 30 real dry-runs passed.

The strict 245-row preflight used same-benchmark/judge observed costs:

- 376 possible new responses
- projected cost $11.577291
- sum of per-pair upper costs $43.891934
- largest pair upper cost $2.973663; runtime 1.5× reserve $4.460494

Immediately before paid calls, key usage remained `2933.143196682`: $51.589830234 above the original baseline `2881.553366448`, leaving $98.410170 under the $150 aggregate cap.

Fresh endpoint snapshot: `provenance/endpoints-20260831T032308Z.json`. The intended routes were live: OpenAI for GPT-4o/GPT-5.2/o4-mini, Google AI Studio for Gemini 2.5 Pro, and DeepInfra fp8 for DeepSeek V4 Flash. Model YAMLs retained `only`, `allow_fallbacks: false`, and `data_collection: deny`; `require_parameters` remained true except the documented GPT-5.2 alias case.

Five disjoint workers ran only the 30 missing configs; all prior 215 successes were skipped before reserve checks/API calls. Final new-call audit:

| Judge | Provider | Calls | Cost |
|---|---|---:|---:|
| DeepSeek V4 Flash | DeepInfra | 76 | $0.193667 |
| Gemini 2.5 Pro | Google AI Studio | 72 | $3.846282 |
| GPT-4o 2024-11-20 | OpenAI | 72 | $4.398897 |
| GPT-5.2 | OpenAI | 84 | $3.459531 |
| o4-mini | OpenAI | 72 | $2.030776 |

Total: 376/376 responses, $13.929153 logged cost, zero served-provider mismatches, zero paid responses missing provider provenance, and zero failed ledger events. The full corpus now contains 3,220 audited calls and $65.5190 logged cost from the original baseline, below the $150 cap.

## RunPod lifecycle and cost

- Pre-task state: no pods; balance $46.143585; preserved volume `c16kxrwrr4`, 50 GB, AP-JP-1.
- Initial pod `ckxd596vw8fbbd`: Secure H100, $3.29/h. It ran the first gates/smokes, was stopped while fixes were made, and was deleted after its host could not resume for lack of capacity.
- Replacement pod `hcutc6jtodrinp`: Secure H200, $4.59/h. It was stopped during local-only changes, then ran trimming, paid top-up, and final analysis.
- Full analysis: 1,000 resamples, 10 spawn workers with BLAS threads fixed to one. Start approximately 03:43 UTC; final outputs approximately 07:02 UTC. The long uncapped tail was preserved rather than reducing resamples; a non-mutating `py-spy` attempt was denied because the container lacked `SYS_PTRACE`, while `/proc` confirmed the worker remained runnable at ~100% CPU with flat memory.
- External deletion watchdogs were used throughout and rearmed for the required long sensitivity/ECI loops.
- Recovered roots passed checksum-only remote/local comparison: analysis, capability baseline, coverage, run provenance, endpoint provenance, and 245 configs. The compact Gemini log hash also matched.
- The ephemeral credential was removed before deletion. Final pod list was empty; the volume remained present.
- Post-delete balance: $25.456021. Observed RunPod balance decrease during Task 5 was about $20.6876, including pod time and ongoing volume billing.

## Scientific diagnostics

- Input rows: 122,779 existing verdicts plus 3,219 latest-success paid verdict rows; 46,748 exact-key deduplicated rows; 26,780 frozen-Q1, exact-epoch, both-correct rows.
- Headline predictions: 240 rows; no missing actuals; 79/80 block coverage.
- Headline fits: 226 successful, 14 failed. Failures are 11 genuinely separated binomial-logit outcomes plus the one frozen missing block repeated across Q2/Q3/Q4.
- Successful-fit maximum absolute weighted gradients: intercept `7.74e-7`, slope `8.71e-7`, both below `1e-6`.
- Zero successful block/quartile groups had all nonzero anchor residuals of one sign.
- No frontier target ID appears in any headline anchor set.
- Primary capped maximum: eight item IDs per unordered pair. Uncapped maximum: 49.
- Successful headline bootstrap count: min 333, median 1,000, max 1,000. Twelve SWE rows are explicitly `bootstrap_unstable`; none are repaired or tuned.
- ECI/pairing: 184 successful and 56 failed rows for both ECI and paired judge sensitivity, with zero `eci_anchor_ids` mismatches.
- ECI failures: 48 rows with only three mapped anchors, three with only two, and five separated fits.
- Block-level ECI diagnostic coverage: 63/80.
- The reviewed mapping has 26 mapped identities and 11 models absent from the Epoch CSV. Only six absent IDs occur in final headline anchor sets: `gemma-3-12b-it`, `granite-4.1-8b`, `ling-2.6-flash`, `qwen3-32b`, `qwen3.5-4b`, `qwen3.5-9b`. They are dropped only in the diagnostic, listed in `dropped_missing_eci_ids`, never imputed.
- MAE, bias, Spearman, inside/outside anchor range, capped/uncapped, and per-model residuals are exported in the named CSVs. Target truth is read only after each anchor-only fit.

## Visual QA

Automated checks:

- 180 judge figure files and 180 ECI figure files;
- all PNGs 605×717 px (about 2.02×2.39 inches at 300 dpi; the axes originate from a 1.75-inch canvas and the outside legend/y-label account for the tight-bbox width);
- every SVG contains editable `<text>`;
- endpoint ticks include 2024/2027 and 0/.5/1;
- identical axes and plot grammar between judge and ECI panels.

Original-resolution visual inspection covered GPQA and SWE Q2/Q3/Q4 across GPT-4o, Gemini 2.5 Pro, GPT-5.2, o4-mini, and DeepSeek V4 Flash, plus matched ECI panels and points variants. Text was readable; legends were outside; release lines, black actuals, blue predictions/bands, and dotted residuals were correct; panel tags did not overlap data; no labels or marks were clipped. No plot or calibrator setting was tuned against target truth.

## Concerns

- The public Gemini 3 SWE source contains only 46 of the current 50 frozen items. The four missing source items remain missing rather than imputed; per-quartile denominators are explicit above.
- One coverage block remains genuinely unidentifiable and 11 headline fits are separated. These failures are preserved as data, not repaired.
- Twelve successful point fits have fewer than 800 successful bootstrap refits and are marked unstable; consumers should use the exported flag.
- Epoch ECI does not contain every canonical anchor. The ECI result is therefore a mapped-subset diagnostic paired with a judge refit on exactly the same IDs, not a replacement headline baseline.
- OpenRouter endpoint precision labels remain provider testimony. The fresh endpoint snapshot and per-response served-provider audit are retained for reproducibility.
