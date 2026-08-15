# Build Monte Carlo simulator of results

## Goal

We are finding that a full data collection would cost a prohibitively large
amount of money. So we'd like to get some estimates on how much data we'd
actually need, and what judges/comparisons we might want to include. This
should be based on all the GPQA data collected in the gpqa-pilot-rd2 and
data_expansion runs. By estimating distributions on judging accuracy, model
performance on evals, intersection of correct items, length bias of judges,
positional bias of judges etc., and how these interact and vary with each
other (and of course, their estimated variability) we should be able to
create a simulator to approaximate how our results may appear given a
particular selection of evaluated models and judging comparisons.

## Output

An interactive dashboard with sliders to set means and variances for every
parameter that we can about, that will update some results plots that we design.

The errorbars should scale with judge counts, comparison counts, and so on.

The primary results plots should be:
1) A basic capability-gap vs judging accuracy curve
2) A scatter plot of easy item-derived judge rankings versus true
   performance on harder items (see irt-based-prediction doc and outcomes)

## Status

Implemented from the reviewed design in
`docs/superpowers/specs/2026-08-14-monte-carlo-cost-simulator-design.md`.

## Run order

```bash
uv run python scripts/irt_proxy/run.py
uv run mc-cost catalogue --config configs/mc_cost/default.json
uv run mc-cost prepare --config configs/mc_cost/default.json
uv run mc-cost fit --config configs/mc_cost/default.json
uv run mc-cost validate --config configs/mc_cost/default.json
uv run mc-cost simulate --config configs/mc_cost/default.json --judges 3 --anchors 8 --comparators 4 --items 100
uv run mc-cost export-dashboard --config configs/mc_cost/default.json
uv run python -m http.server 8000 -d interactive_viewer
```

The dashboard is at `http://localhost:8000/mc_cost.html`. The browser only reads
precomputed results; it does not fit or approximate the model.

## Inputs and reruns

- `configs/mc_cost/default.json` selects all source paths and holds fit,
  simulation, and dashboard grids. Automatic `J`, `M`, and `N` can also be set
  with `--judges`, `--anchors`, and `--comparators` on `simulate`.
- `configs/mc_cost/model_overrides.json` is the durable MC-specific metadata
  input. Edit it to declare identity aliases, automatic judge/candidate pool
  membership, observed-role flags, or a deliberate metadata correction.
- `docs/canonical-models.md` seeds stable canonical IDs. Fitted fixed-3PL GPQA
  models are unioned into the catalogue, so this file does not freeze the
  future catalogue.
- `model-sampler/model_pool.json` supplies release, family, reasoning-class,
  and price metadata when names resolve. It does not decide MC pool membership;
  pool flags come from the explicit overrides file.
- `irt_results/easy_item_prediction/tables/fit_capabilities.csv`,
  `irt_results/easy_item_prediction/tables/fit_difficulties.csv`, and
  `irt_results/easy_item_prediction/posterior/irt_posterior.npz` are products of
  `scripts/irt_proxy/run.py`, not hand-edited MC inputs.
- `logs/judging-variations/gpqa/a_baseline` supplies the observed pairwise
  judgments and judge token usage. `data/gpqa` and
  `external_data/epoch-gpqa` supply candidate token observations.

After changing canonical IDs, model-pool metadata, overrides, or IRT products,
rerun `catalogue`, `prepare`, `fit`, `validate`, `simulate`, and
`export-dashboard` in order. After changing judgment or candidate logs, rerun
from `prepare`; after changing fit settings, rerun from `fit`; after changing
only simulation/dashboard grids or manual triples, rerun `simulate` as needed
and then `export-dashboard`. Generated files in `analysis/mc_cost` are
hash-bound outputs and must not be edited.

The MC preparation path resolves observed aliases to catalogue IDs and uses
only uncertainty-backed fixed-3PL capability values. It does not insert legacy
raw GPQA accuracy as `theta`. Models without capability or release metadata
remain visible in the catalogue but are excluded from automatic pools and from
feature-dependent training rows; their manual scenarios report missing support
instead of receiving an average-model substitution.

The baseline judgment protocol uses `include_reasoning=false`. Candidate-token
preparation therefore fits and simulates the final answer length visible to the
judge, while retaining full transcript length and generated-reasoning evidence
for audit and future protocols. Each visible transcript is capped at the
baseline's 65,536-character limit; paired-truncation conditions are unsupported
and rejected. Judgment artifacts record this presentation condition and fail
closed if it does not match. The current judgment data also
contain no self-comparisons or reasoning-class contrasts, so those effects are
excluded from the sanctioned models; explicit manual scenarios receive named
support warnings until matching observations exist.

`prepare` scans every Inspect archive under both configured candidate roots and
can take several minutes; it publishes its generation atomically only after the
scan completes. Its manifest-bound `analysis/mc_cost/prepare_accounting.json`
records per-judge extraction counts and eligibility plus named, aggregated
judgment/candidate exclusions; inspect this file before fitting. The Task 11
production preflight completed catalogue and judgment preparation and checked
representative candidate archives from both roots, rather than running the
complete candidate-token scan or any production fit.
