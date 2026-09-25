"""Throwaway previews of every option combination of the three results figures, as PNGs in
analysis/preview/, so the layouts to keep can be pointed at. Reads a forecasts table from
disk (never downloads anything), reuses analysis.stats and plots.results unchanged.

Cell script: `PYTHONPATH=src uv run python scripts/analyses/preview_plots.py` from the repo
root, or cell by cell (`# %%`) after the cells above. Writes analysis/preview/index.md.
"""

# %% settings
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis import stats  # noqa: E402
from plots import results as plots  # noqa: E402
from analysis.stats import BASELINE, BEST, KEYS, MEAN, one_per_target, task_metrics  # noqa: E402

OUT = Path("analysis/preview")
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = ["analysis/results/forecasts.csv", "analysis/results-smoke/forecasts.csv"]  # first on disk wins
DEFAULT = {"pairwise": "linear", "direct": "raw", "embedding": "full"}
# the default toggles, then one set per alternative variant (bar plots and tables only)
VARIANT_SETS = {"": DEFAULT, "_pairwise-calibrated": {**DEFAULT, "pairwise": "calibrated"},
                "_direct-item-z": {**DEFAULT, "direct": "item_z"}, "_embedding-answer": {**DEFAULT, "embedding": "answer"}}
DOMAINS = ["science", "math", "coding", "cyber"]
LATEST = "2026-I"
METRICS = ["mse", "delta", "z", "z_baseline"]
plots.METRIC_LABELS["z_baseline"] = "MSE − easy-item MSE\n(easy-item bootstrap sd)"
index, skipped = [], []  # (file name, description) / (what, why)

# %% load: the forecasts table (new results table, else the smoke table adapted to its columns)
INPUT = next(p for p in INPUTS if Path(p).exists())
f = pd.read_csv(INPUT)
if "variant" not in f.columns:  # the smoke table: earlier protocols, no variants, no reference LOO error
    f = f[f.protocol.isin(["reference_mean", "accuracy_all", "glm_regex", "glm_length", "pairwise", "direct"])]
    f = f.replace({"protocol": {"reference_mean": BASELINE, "accuracy_all": "accuracy_hard", "glm_length": "embedding"}})
    # no reference LOO error in the old table: rank judges by name so "best judge on references" picks one deterministically
    f = f.assign(variant="-", judge_loo_mse=np.where(f.judge == "-", np.nan, f.judge.rank(method="dense")))
    VARIANT_SETS = {"": {}}
f = f.replace({"protocol": {"intercept": BASELINE}})  # older tables key the baseline "intercept"
present = [d for d in DOMAINS if d in set(f.domain)]
print(f"input {INPUT}: domains {present}, semesters {sorted(set(f.semester))}")


def save(name: str, description: str) -> Path:
    index.append((name, description))
    return OUT / name


# %% figure 1: true theta vs forecast dots
for protocol in ["pairwise", "direct"]:  # forecast dots only for the judge conditions
    variant = VARIANT_SETS[""].get(protocol, "-")
    if not ((f.protocol == protocol) & (f.variant == variant)).any():
        skipped.append((f"forecasts_{protocol}_*", f"no rows for {protocol}/{variant} in the input table"))
        continue
    for sem_tag, semesters in [("all", None), (LATEST, [LATEST])]:
        # pooled: pick each domain's best judge first, then tag models by domain so nothing is deduplicated
        pooled = one_per_target(f)
        pooled = pooled.assign(model=pooled.model + " (" + pooled.domain + ")", domain="all", judge_loo_mse=0.0,
                               condition=np.where(pooled.judge == "-", "-", "reasoning-shown"))  # re-selection keeps every row
        views = [("domains", f), ("pooled", pooled)] + [(d, f[f.domain == d]) for d in DOMAINS]
        for view, table in views:
            if table.empty:
                skipped.append((f"forecasts_{protocol}_{sem_tag}_{view}.png", f"domain {view} not in the input table"))
                continue
            path = save(f"forecasts_{protocol}_{sem_tag}_{view}.png",
                        f"{plots.LABELS[protocol]} ({variant}); semesters: {sem_tag}; panels: {view}")
            plots.forecast_figure(table, protocol, variant, path, semesters=semesters)

# %% figure 2: bars against the easy-item prediction


def baseline_spread(forecasts: pd.DataFrame, seed: int = 0, draws: int = 2000) -> pd.DataFrame:
    """sd of the easy-item prediction's bootstrap MSE (targets resampled with replacement) for every
    task_metrics group, on the targets the condition forecast. Columns: the task_metrics keys
    plus `baseline_sd`."""
    frame = one_per_target(forecasts)
    base = frame[frame.protocol == BASELINE][["domain", "semester", "model", "theta_hat", "theta"]]
    base = base.assign(err_baseline=(base.theta_hat - base.theta) ** 2)[["domain", "semester", "model", "err_baseline"]]
    paired = frame.dropna(subset=["theta_hat"]).merge(base, on=["domain", "semester", "model"])
    rng = np.random.default_rng(seed)

    def sd(g):
        e = g.err_baseline.to_numpy(float)
        return float(rng.choice(e, size=(draws, len(e))).mean(axis=1).std(ddof=1))

    rows = []
    for k, g in paired.groupby(["semester", *KEYS], sort=False):
        rows.append(dict(zip(["semester", *KEYS], k), baseline_sd=sd(g)))
    for k, g in paired.groupby(KEYS, sort=False):
        rows.append(dict(zip(KEYS, k), semester="all", baseline_sd=sd(g)))
    for k, g in paired.groupby(KEYS[1:], sort=False):
        rows.append(dict(zip(KEYS[1:], k), domain="all", semester="all", baseline_sd=sd(g)))
    return pd.DataFrame(rows)


def metrics_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """task_metrics plus z_baseline = (mse − easy-item MSE) / sd of the easy-item bootstrap MSE (not paired)."""
    m = task_metrics(forecasts).merge(baseline_spread(forecasts), on=["domain", "semester", *KEYS[1:]], how="left")
    return m.assign(z_baseline=(m.mse - m.baseline_mse) / m.baseline_sd)


def mean_over_judges(forecasts: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """The best-judge metrics table with the judge protocols' rows replaced by the mean over
    judges: each judge's rows alone through metrics_table, then the mean of every metric per
    (domain, semester, protocol, variant); judge = MEAN, condition "core"."""
    judges = sorted(set(forecasts.judge) - {"-"})
    per_judge = pd.concat([metrics_table(forecasts[(forecasts.judge == "-") | (forecasts.judge == j)]) for j in judges])
    per_judge = per_judge[per_judge.protocol.isin(stats.JUDGED)]
    values = ["n", "mse", "baseline_mse", "delta", "z", "z_baseline", "baseline_sd"]
    mean = per_judge.groupby(["domain", "semester", "protocol", "variant"], as_index=False)[values].mean()
    return pd.concat([best[~best.protocol.isin(stats.JUDGED)], mean.assign(judge=MEAN, condition="core")], ignore_index=True)


best_metrics = metrics_table(f)
judge_tables = {"best-judge": best_metrics, "mean-judges": mean_over_judges(f, best_metrics)}
for tag, variants in VARIANT_SETS.items():
    for metric in METRICS:
        for judge_tag, metrics in judge_tables.items():
            for colour in ["black", "by-domain"]:
                path = save(f"bars_{metric}_{judge_tag}_{colour}{tag}.png",
                            f"metric {metric}; judge rows: {judge_tag.replace('-', ' ')}; points {colour}; variants {variants or '-'}")
                plots.bar_figure(metrics, metric, variants, path, by_domain=colour == "by-domain")
# single-semester versions: default variants, best judge on references; the points are the domains
for metric in METRICS:
    for sem in sorted(set(f.semester)):
        for colour in ["black", "by-domain"]:
            path = save(f"bars_{metric}_best-judge_{colour}_{sem}.png",
                        f"metric {metric}; semester {sem} only (points = domains); judge rows: best judge; points {colour}; variants {DEFAULT}")
            plots.bar_figure(best_metrics, metric, DEFAULT, path, by_domain=colour == "by-domain", semesters=[sem])

# %% figure 3: semester x condition tables


def table_png(table: pd.DataFrame, path) -> None:
    """The table as an 8pt matplotlib table, no frame, 2 decimals, NaN as an en dash."""
    text = table.map(lambda v: "–" if pd.isna(v) else f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(1.1 * (len(table.columns) + 1), 0.18 * (len(table) + 1)))
    ax.axis("off")
    t = ax.table(cellText=text.to_numpy(), rowLabels=table.index, colLabels=table.columns, loc="center", edges="open")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(range(len(table.columns)))
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


for tag, variants in VARIANT_SETS.items():
    for metric in METRICS:
        for scope in ["pooled", *DOMAINS]:
            if scope != "pooled" and scope not in present:
                skipped.append((f"table_{metric}_{scope}{tag}.{{csv,png}}", f"domain {scope} not in the input table"))
                continue
            # semester_table asserts metric in stats.METRICS: hand it z_baseline under the name "z"
            m = best_metrics.assign(z=best_metrics.z_baseline) if metric == "z_baseline" else best_metrics
            table = stats.semester_table(m, "z" if metric == "z_baseline" else metric, variants, None if scope == "pooled" else scope)
            path = save(f"table_{metric}_{scope}{tag}", f"metric {metric}; scope {scope}; best judge on references; variants {variants or '-'}")
            table.to_csv(path.with_suffix(".csv"))
            table_png(table, path.with_suffix(".png"))

# %% index
lines = [f"# Preview index\n", f"Input table: `{INPUT}` (domains present: {', '.join(present)}).",
         "Variant sets (file-name suffix: variant per protocol; the empty suffix is the default set):"]
lines += [f"- `{tag or '(none)'}`: {variants or '-'}" for tag, variants in VARIANT_SETS.items()]
if len(present) == 1:
    lines.append(f"\nOnly {present[0]} is in the input table, so the `domains` and `pooled` forecast views show the same "
                 "single panel as the domain's own file (the pooled panel is tagged \"all\").")
lines += ["\n## Files\n"]
lines += [f"- `{name}` — {desc}" for name, desc in index]
lines += ["\n## Not rendered\n"] + [f"- `{name}`: {why}" for name, why in skipped]
(OUT / "index.md").write_text("\n".join(lines) + "\n")
print(f"{len(index)} outputs, {len(skipped)} skipped -> {OUT}/index.md")

# %% contact sheet: every PNG on one scrollable page, grouped by figure family
html = ["<html><body style='font:13px Arial;margin:16px'><h2>Preview contact sheet</h2>",
        f"<p>Input: {INPUT}. Skipped {len(skipped)} combinations (see index.md).</p>"]
for family, title in [("forecasts_", "Figure 1: true θ vs forecast"), ("bars_", "Figure 2: bars against the easy-item prediction"),
                      ("table_", "Figure 3: semester × condition tables")]:
    html.append(f"<h3 id='{family}'>{title}</h3><div style='display:flex;flex-wrap:wrap;gap:16px'>")
    for name, desc in index:
        if name.startswith(family):
            png = name if name.endswith(".png") else name + ".png"
            html.append(f"<figure style='margin:0;width:{'100%' if family == 'forecasts_' else '380px'}'>"
                        f"<img src='{png}' style='max-width:100%'><figcaption><b>{png}</b><br>{desc}</figcaption></figure>")
    html.append("</div>")
(OUT / "index.html").write_text("\n".join(html) + "</body></html>\n")
print(f"contact sheet -> {OUT}/index.html")
