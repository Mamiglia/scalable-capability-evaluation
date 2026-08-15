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
uv run mc-cost simulate --config configs/mc_cost/default.json --judges 4 --anchors 8 --comparators 4 --items 100
uv run mc-cost export-dashboard --config configs/mc_cost/default.json
uv run python -m http.server 8000 -d interactive_viewer
```

The dashboard is at `http://localhost:8000/mc_cost.html`. Edit durable model
metadata in `model-sampler/model_pool.json`, then rerun `catalogue`, `prepare`,
`fit`, `validate`, and `export-dashboard`; the generated catalogue and fitted
artifacts are hash-bound, and the browser does not fit or approximate the model.
