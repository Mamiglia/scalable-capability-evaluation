# Handoff: fit the full HiBayes GLM on the new HF GPQA data

## Correction

The 2026-08-17 HF fit in commits `b0dfbc8`, `48ab31b`, and `9623b3d` is **not the requested analysis**.

It made two scientific errors:

1. It pooled GPQA and MBPP. MBPP must be excluded.
2. It fit only `length_difference_sd`, `capability_difference_sd`, and
   `same_model_family`. It omitted the main GLM's stylistic predictors.

Do not cite its coefficient, LOO scores, or figures. The code and artifacts are
useful only as provenance for what went wrong.

## Where the correct full GLM lives

The full model is already implemented. Do not invent another reduced NumPyro
model.

- `scripts/hibayes/model.py`: hierarchical Bernoulli left-choice model with
  partially pooled judge slopes, judge position intercepts, condition effects,
  and item effects.
- `scripts/hibayes/config.yaml`: authoritative full predictor list and sampler
  settings (4 chains, 2,000 warmup, 2,500 draws, target accept 0.95).
- `scripts/hibayes/prepare_from_verdicts.py`: authoritative left/right
  orientation, style-difference joining, reasoning filter, and standardisation
  logic for the older exported-verdict data.
- `scripts/hibayes/style_features.py` and `style_lexicon.yaml`: exact style
  feature definitions.
- `scripts/hibayes/style_rates.py`: older-data extraction of per-transcript
  style rates.
- `scripts/hibayes/communicators.py`: the main GLM effect tables, PSIS-LOO, and
  plots.
- `scripts/hibayes/run.sh`: normal full-pipeline entry point.

The authoritative predictors in `config.yaml`, in order, are:

1. `length_difference_sd`
2. `capability_difference_sd`
3. `same_model_family`
4. `self_comparison`
5. `personal_pronouns_difference_sd`
6. `hedging_difference_sd`
7. `backtracking_difference_sd`
8. `uncertainty_difference_sd`
9. `confidence_difference_sd`
10. `questions_difference_sd`

## Correct HF data path

Build a new HF-specific frame adapter, but feed the existing model and full
predictor list.

1. Read only `data/hf-judging-2026-08-17/raw/pilot-*-gpqa-*`; reject MBPP at
   discovery time and assert `frame["bench"].unique() == ["gpqa"]` before fit.
2. Keep the standing Phi-judge exclusion from
   `scripts/analyses/hf_judging_round.py::keep_judge`.
3. Read successful, non-invalidated evals named by each directory's
   `logs.json`. Deduplicate exact artifacts only; retain distinct response
   epochs.
4. For every sample, use
   `scripts/analyses/hf_judging_viewer.py::parse_prompt` to recover the exact two
   transcripts shown to the judge. Do not substitute aggregate `logs.json`
   metrics or a different source transcript.
5. Preserve orientation exactly: `sample.target` is the slot containing the
   major model and `sample.scores["judge_scorer"].answer` is the chosen slot.
   Every feature is left minus right.
6. Run `style_features.feature_rates` on each exact shown transcript using
   `style_lexicon.yaml`. Add left-minus-right differences for all six style
   categories.
7. Compute transcript `reasoning_chars` with the same answer-stripping rule as
   the main pipeline. Apply `configs/reasoning_filter.yaml` (currently
   per-model, 100-character floor, exclude low/high effort variants) before
   calculating standardisation scales.
8. Standardise length, capability, and all six style differences on the final
   retained GPQA rows. Keep `same_model_family` and `self_comparison` binary.
9. Use one transcript condition such as `hf_gpqa`; do not encode MBPP as a
   condition.
10. Fit with the existing `judge_feature_logistic_model` and the exact sampler
    settings from `config.yaml`.

Capability scale must be explicit in the new config. The existing older-data
pipeline defaults to ECI, while the HF gap-curve analysis uses pooled GPQA
benchmark accuracy. If the goal is literal replication of the main GLM, use
ECI; if it is replication of the HF capability-gap result, use GPQA accuracy.
Do not silently mix the two or pool benchmark-specific scales.

## Required checks before submission

- A test where swapping transcript slots negates every continuous/style
  difference and flips `chosen_left`.
- A test proving an MBPP directory contributes zero rows.
- A test proving all ten configured predictors exist, are finite, and are in
  the same order in preprocessing, model config, and communication labels.
- Audit counts for excluded judges, missing capability scores, unparsed picks,
  truncated prompts, reasoning-filtered models/rows, and exact duplicates.
- Frame report: rows, GPQA items, judges, candidate models, model pairs, and
  response epochs.
- Four-chain diagnostics: zero divergences, R-hat, bulk/tail ESS, and PSIS
  Pareto-k counts.

## Invalid artifacts from the wrong run

- Local frame: `analysis/hibayes/hf_2026_08_17.csv`
- Local posterior/results: `analysis/hibayes/run_hf_2026_08_17_v3/`
- SWC runs:
  `/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/hibayes_hf_20260817_v{1,2,3}`
- Completed wrong-model jobs: `3391822` and `3392092`
- Failed/cancelled plot jobs: `3392093` and `3392165`
- Preserved failed plot temp directory:
  `.../hibayes_hf_20260817_v3/plots.tmp-3392093`

Ceph policy forbids deleting or overwriting these. Leave them in place unless
the user explicitly performs cleanup.

## Repository state

The wrong-run commits are interleaved with other active work on branch
`analysis`; do not reset or revert the branch wholesale. Replace or revert only
the affected HiBayes files after inspecting concurrent changes:

- `b0dfbc8` — HF converter (orientation logic reusable, dataset/features wrong)
- `48ab31b` — reduced standalone GLM/SWC runner (wrong model)
- `9623b3d` — plots of the wrong posterior

Unrelated `.gitignore` and `judge-whitebox` changes belong to other work and
must remain untouched.
