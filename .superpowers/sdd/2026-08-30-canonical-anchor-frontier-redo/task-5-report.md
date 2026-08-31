# Task 5 report: canonical forecasts, paired ECI baseline, and release-time figures

> **Superseded main-analysis notice (Fix Round 1):** The final user-facing result is now the deliberately in-sample rank–accuracy analysis at `analysis/rank_regression/`. The anchor forecasts and ECI baseline documented in the original report below are retained only as discarded/non-main provenance. The final specification and evidence are appended under **Fix Round 1**.

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

## Fix Round 1: superseding in-sample rank–accuracy analysis

### Final scientific contract

The final main analysis deliberately does not estimate a held-out forecast. For each of the five judges and two benchmarks it fits one Bradley–Terry ranking from all 26,780 frozen-Q1, exact-epoch, both-correct protocol rows, after the primary deterministic cap of eight item IDs per unordered pair. There is no answer-model release cutoff and no target-specific rank refit.

For each judge×benchmark ranking and each of Q2/Q3/Q4, every model with both BT ability and observed hard-quartile accuracy enters the in-sample curve fit. The two equal-parameter candidates are:

- free-intercept ordinary least squares, computed by `np.linalg.lstsq([1, ability], accuracy)`;
- free-intercept bounded-response sigmoid, `logistic(intercept + slope * ability)`, fit deterministically to binomial hard-outcome counts.

Both candidates export coefficients, model-level SSE, and model-level RMSE. Selection is deterministic: sigmoid when any linear training fitted value is outside [0,1], or when sigmoid RMSE is strictly lower; linear otherwise. All fitted values are raw in the CSVs. Only plot display values are clipped to [0,1] if necessary. The exact canvas/export label is `Rank–accuracy fit (in-sample)`; the final tables and figures use neither forecast nor held-out language.

The unambiguous main-output manifest is `judge_results/frontier/eci-semiannual-canonical-anchors/MAIN_OUTPUT.json`. It points to `analysis/rank_regression/` and marks the earlier `analysis/figures/frontier_style` and `capability_baseline/figures` roots discarded/non-main. ECI and anchor subsets do not enter the final main analysis.

### Review defects fixed before the final ruling

Before the main-analysis specification was superseded, the review correctly found target leakage in the anchor bootstrap implementation. The corrected code and retained regressions now ensure:

- headline, uncapped, paired-judge, and old ECI-sensitivity point/bootstrap frames contain selected-anchor rows only;
- changing target correctness and target-only item availability leaves point estimates, intervals, coefficients, successful-refit counts, and instability flags identical, changing only actual/residual values;
- every requested selected anchor must exist in both ability and hard outcomes; a five-requested/four-present case fails explicitly;
- fitted and residual anchor IDs exactly equal the requested IDs and order;
- failed interior plotted values remain NaNs on the full release grid, so lines and fills break rather than bridge;
- actual markers at 0 and 1 use `clip_on=False` while ticks remain 0/.5/1;
- the discarded bootstrap implementation schedules deterministic one-block×quartile tasks dynamically and writes atomic phase markers.

These fixes remain tested because they prevent the discarded sensitivity code from silently producing invalid results, but none of those anchor/ECI artifacts is the final main output.

### RED/GREEN evidence

The first review RED run produced seven expected failures across leakage invariance, missing selected anchors, residual ordering, ECI leakage, and plot-gap/clipping regressions. Scheduler/checkpoint REDs additionally showed missing singleton-task and atomic-marker interfaces. After implementation, the affected suite was green and the full local suite reached 201 passing tests.

The final rank–accuracy ruling began with an eight-failure RED because `rank_regression.py` did not exist. The focused GREEN verifies exact OLS coefficients, all-model joins including frontier targets, intentional target-truth influence, method selection, Q2/Q3/Q4 coverage, absence of ECI inputs, missing-ability actual retention, and the exact plot label:

```text
$ uv run pytest tests/frontier_eci/test_rank_regression.py tests/frontier_eci/test_analysis.py -q
18 passed
```

The broader affected local gate passed 55 tests; compilation and `git diff --check` exited zero. The same focused gate passed 18/18 on RunPod before the full-data execution.

Final fresh repository-wide verification passed `211 passed` with one pre-existing Pydantic deprecation warning. Six multiprocessing tests initially failed inside workspace isolation because the sandbox denied their local manager socket; rerunning the identical suite with local IPC permission passed all 211 tests.

### Final outputs and diagnostics

`analysis/rank_regression/` contains:

- `abilities.csv`: 234 model abilities from ten judge×benchmark BT rankings;
- `ranking_blocks.csv`: ten block audits, including verdict counts and maximum pair-item count;
- `fits.csv`: 30 Q2/Q3/Q4 two-candidate fits with equations, coefficients, SSE/RMSE, selection, and exact training IDs;
- `training_fitted.csv`: every joined in-sample model point with both candidate fitted values and residuals;
- `frontier_fitted.csv`: the complete 240-row judge×benchmark×quartile×frontier grid;
- `summary.json` and atomic `checkpoint-main.json`;
- `figures/`: 30 line stems and 30 points stems, each as PNG/PDF/editable-text SVG (180 files total).

Full-data audit:

- 10/10 rankings and 30/30 curve-fit blocks succeeded;
- the cap audit maximum is exactly eight item IDs per unordered pair;
- every fit's exported training IDs exactly equal its joined model rows;
- 18–31 models enter each fit (median 23.5);
- independently recomputed `np.linalg.lstsq` coefficients and all 30 selection decisions exactly match exports;
- selected methods: 23 sigmoid, 7 linear; 21 sigmoid choices were forced by out-of-range linear training fits and two by lower sigmoid RMSE;
- linear RMSE range 0.0896–0.2729 (median 0.1717); sigmoid RMSE range 0.0743–0.2775 (median 0.1686);
- 240/240 frontier rows have actuals and fitted values across all eight frontier models;
- after removing target-specific cutoffs, the pooled all-Q1 design yielded zero frontier BT-ability gaps. The earlier anticipated single gap therefore does not exist under the final specification and was not fabricated;
- Gemini 3 Pro SWE-mini uses the explicit source denominators Q2/Q3/Q4 = 13/12/9, with accuracies 0.846/0.417/0.111.

### Final RunPod lifecycle and visual QA

Fix Round 1 used Secure H100 pod `a74kw8pd94zkuk` at $3.29/h on preserved AP-JP-1 volume `c16kxrwrr4`. It was provisioned only after the first local gate. A superseded 1,000-bootstrap run was terminated at headline task 77/240 when the main-analysis ruling changed; the pod was stopped during local-only work. It resumed at 08:00:49 UTC after the 55-test local gate. The final rank-regression run completed in about 86 seconds, then artifacts were recovered and checksum-verified before deterministic end-of-line whitespace normalization of generated SVG path data:

```text
remote/local combined SHA-256:
dafbef21d5fbfa12fd9c817bfe0d954e311c08b65088ad371f50a5a3167e0b2b
```

The normalized committed main root has combined SHA-256 `2e2d74ba242dfa1cf194ff878db15882c6c44daa49c27e634e2b86908dbd792d`; normalization changes no pixels, geometry, or editable text.

The pod was deleted immediately afterward; `runpodctl pod list` returned `[]`. The network volume remains. Final balance was $23.643660; the Fix Round 1 balance decrease from its pre-pod checkpoint was approximately $1.44, including volume billing.

Automated visual audit found 60 PNG, 60 PDF, and 60 editable-text SVG files. Every PNG is 687×717 px at 300 dpi; all SVGs contain the exact in-sample label and editable `<text>`. There are 20 PNG panels for each of Q2/Q3/Q4 and equal GPQA/SWE-mini coverage.

Original-resolution inspection covered both benchmarks and all three quartiles across GPT-4o, Gemini 2.5 Pro, GPT-5.2, o4-mini, and DeepSeek V4 Flash, including line and points variants. Legends are outside and readable; grey panel tags do not overlap; judge-release lines, black actuals, blue fitted values, and dotted residuals are visible; endpoint markers at 0/1 are not clipped; axes/ticks are identical across panels. No uncertainty band is drawn or implied.

### Final concerns

- This is intentionally descriptive and in-sample. Its RMSE and residuals do not estimate out-of-sample forecasting performance.
- Model selection uses the same in-sample rows as fitting, exactly as specified.
- The existing BT implementation uses its fixed ridge of 0.01 to keep unbeaten model abilities finite; the rank-regression layer adds no further regularization.
- The four absent upstream Gemini SWE items remain absent; the explicit per-quartile denominators above are used without imputation.
