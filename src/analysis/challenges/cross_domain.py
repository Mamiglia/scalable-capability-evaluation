"""Transcripts from one domain, calibrated on another domain's theta*.

Can a protocol forecast capability in one domain from transcripts collected in another, when the
references' capability in the scored domain is known? For each semester and each ordered pair of
domains, the source domain's task is rebuilt with every theta* replaced by the scored domain's for
the same semester: the references' and historical models' theta*, which is what each protocol fits
to, and the targets' truth, which is what the forecast is scored against. Items, transcripts,
judge runs and their matching stay the source domain's, so `calibration.forecasts` runs unchanged:

- Pairwise judge: verdicts on the source domain's items, references pinned at their scored-domain
  theta* in the linear conversion.
- Absolute scoring: OLS of the references' scored-domain theta* on their mean source-domain rating.
- Embedding and Deterministic features: regressions from source-domain transcript features to
  scored-domain theta*, fitted on references and history.
- Each judge protocol keeps, in every cell of a row, the judge that is best on references within
  the source domain itself, figure one's choice, so a row compares like with like
  (`fixed_judge_rows`).
- The saturated-item prediction is the references' mean scored-domain theta*, identical to that
  domain's own baseline, so a column of the matrix shares its baseline and the diagonal reproduces
  the forecasts in analysis/results/forecasts.csv. The runner asserts that.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import calibration, loading, matching
from analysis.challenges import generalisation as gen
from analysis.challenges.signal_glm import COLORS, DOMAINS, LABELS, PROTOCOLS, score, stars
from analysis import stats
from analysis.plots import DOMAIN_LABELS, chosen, ci95

DOMAIN_LABELS = {**DOMAIN_LABELS, "math": "Maths"}  # the manuscript spells the domain with an s
from analysis.stats import BASELINE
from capability.task import Cohort, Task
from reference import tables


def scored_theta(domain: str, semester: str) -> pd.Series:
    """Every model's theta* on one task of `domain`, from the frozen table; models without one
    are dropped."""
    models, _ = tables.load(loading.domain(domain))
    return models[f"{semester}_theta"].dropna()


def relabel(cohort: Cohort, theta: pd.Series, truth: bool = True) -> Cohort:
    """`cohort` keeping only the models `theta` labels, with their theta* replaced by it (NaN for
    a test cohort, whose theta is unknown by construction). A transcript cache already read stays
    with the new cohort, so swapping theta* costs no rereading of the logs."""
    models = cohort.models[cohort.models.index.isin(theta.index)].copy()
    models["theta"] = theta.reindex(models.index) if truth else np.nan
    swapped = Cohort(models, cohort.items, cohort.semester)
    if "transcripts" in cohort.__dict__:  # cached_property, keyed by model
        swapped.__dict__["transcripts"] = {m: t for m, t in cohort.transcripts.items() if m in models.index}
    return swapped


def swap(task: Task, theta: pd.Series) -> Task:
    """`task` with every theta* replaced by `theta`: the references and the historical cohort that
    protocols fit to, and the targets' truth that the forecast is scored against. Models `theta`
    does not label are dropped from each cohort."""
    return Task(task.semester, relabel(task.references, theta), relabel(task.historical, theta),
                relabel(task.test, theta, truth=False), task.truth.index.to_series()
                .map(theta).dropna().rename("theta"))


def forecasts(domains: list[str], semesters: tuple[str, ...]) -> pd.DataFrame:
    """One row per (source domain, scored domain, semester, target, protocol, variant, judge),
    the columns of analysis/results/forecasts.csv plus `scored_domain`; `domain` is the source.
    Judge runs and transcripts are read once per (source domain, semester)."""
    frames = []
    for source in domains:
        dom = loading.domain(source)
        protocols = calibration.transcript_protocols(dom)
        for task in loading.tasks(dom, list(semesters)):
            runs = [matching.match(r, task) for r in loading.judge_runs(dom, task)]
            for scored in domains:
                theta = scored_theta(scored, task.semester)
                swapped = swap(task, theta)
                if swapped.truth.empty or swapped.references.models.empty:
                    print(f"cross_domain: {source} -> {scored} {task.semester}: no shared labels, skipped")
                    continue
                frame = calibration.forecasts(dom, swapped, protocols, runs)
                frames.append(frame.assign(scored_domain=scored))
                print(f"cross_domain: {source} -> {scored} {task.semester}: {len(frame)} rows, "
                      f"{len(swapped.truth)} targets, {len(swapped.references.models)} references")
    return pd.concat(frames, ignore_index=True)


def fixed_judge_rows(table: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """One forecast per (source domain, scored domain, semester, target, protocol), with its squared
    error `err` and the saturated-item prediction's on the same target `base`. Transcript protocols
    are taken as they are; a judge protocol uses, in every cell of a row, the judge that is best on
    references within the source domain itself (the diagonal cell, which is figure one's choice), so
    a row compares like with like instead of re-picking a judge for each scored domain."""
    core = chosen(stats.core_condition(table), variants)
    core = core[core.protocol.isin(PROTOCOLS)]
    judged = core.protocol.isin(stats.JUDGED)
    key = ["domain", "semester", "protocol", "variant"]
    diagonal = core[judged & (core.domain == core.scored_domain)]
    best = diagonal[diagonal.groupby(key).judge_loo_mse.transform("min") == diagonal.judge_loo_mse]
    best = best.drop_duplicates(key)[[*key, "judge", "condition"]]
    rows = pd.concat([core[~judged], core[judged].merge(best, on=[*key, "judge", "condition"])], ignore_index=True)
    rows = rows.dropna(subset=["theta_hat", "theta"]).assign(err=lambda r: (r.theta_hat - r.theta) ** 2)
    cell = ["domain", "scored_domain", "semester", "model"]
    base = rows[rows.protocol == BASELINE].set_index(cell).err
    return rows.assign(base=rows.set_index(cell).index.map(base)).dropna(subset=["base"])


def cell_scores(table: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """Prediction score per (source domain, scored domain, protocol): the targets' squared errors
    pooled over every semester before scoring, as everywhere else in this analysis, with each judge
    protocol fixed to its source domain's best judge (`fixed_judge_rows`)."""
    rows = fixed_judge_rows(table, variants)
    pooled = rows.groupby(["domain", "scored_domain", "protocol"]).agg(n=("err", "size"), mse=("err", "mean"),
                                                                        baseline_mse=("base", "mean")).reset_index()
    return pooled.rename(columns={"domain": "source", "scored_domain": "scored"}).assign(
        score=lambda d: score(d.mse, d.baseline_mse))


def model_scores(table: pd.DataFrame, variants: dict[str, str], cells: list[tuple[str, str]]) -> pd.DataFrame:
    """One prediction score per target model and protocol over `cells` (source, scored) pairs, the
    squared errors pooled over the cells and semesters first, judges fixed as in `fixed_judge_rows`."""
    rows = fixed_judge_rows(table, variants)
    rows = rows[[(s, d) in cells for s, d in zip(rows.domain, rows.scored_domain)]]
    pooled = rows.groupby(["model", "protocol"])[["err", "base"]].mean().reset_index()
    return pooled.assign(score=score(pooled.err, pooled.base))


def relative_to_within(grid: np.ndarray) -> np.ndarray:
    """Each cell minus its column's diagonal: the score of transcripts from the row's domain against
    the within-domain forecast of the same scored domain. On the log scale that difference is
    itself a prediction score with the within-domain forecast as the baseline: 0 on the diagonal,
    -1 means twice the within-domain forecast's error, +1 half of it."""
    return grid - np.diag(grid)[None, :]


def heatmap_panel(ax, grid: np.ndarray, reach: float, title: str, label_rows: bool) -> "plt.AxesImage":
    """One 4 x 4 matrix on a diverging scale centred on 0 and reaching `reach`, each cell printed."""
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-reach, vmax=reach)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(grid[i, j]) > 0.6 * reach else "black")
    ax.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=90, fontsize=6)
    ax.set_yticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS] if label_rows else [], fontsize=6)
    ax.set_title(title, fontsize=7, color="grey", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    return im


def heatmap_figure(scores: pd.DataFrame, protocol: str) -> plt.Figure:
    """One protocol, two 4 x 4 panels with rows the domain the transcripts come from and columns the
    domain whose theta* the protocol is calibrated on and scored against. Left: the prediction score
    against the saturated-item prediction, in halvings of the error; the diagonal is the ordinary
    within-domain forecast. Right: the same cells against the within-domain forecast of their
    column (`relative_to_within`), so the diagonal is 0 and a negative cell is what changing the
    transcripts' domain costs. Each panel has its own diverging scale centred on 0."""
    grid = (scores[scores.protocol == protocol].pivot_table(index="source", columns="scored", values="score")
            .reindex(index=DOMAINS, columns=DOMAINS).to_numpy(float))
    relative = relative_to_within(grid)
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.0), layout="constrained")
    for ax, values, title, rows in ((axes[0], grid, "vs saturated items", True),
                                    (axes[1], relative, "vs within-domain forecast", False)):
        im = heatmap_panel(ax, values, max(0.5, float(np.nanmax(np.abs(values)))), title, rows)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Transcripts from", fontsize=7)
    fig.supxlabel("Calibrated and scored on", fontsize=7)
    fig.suptitle(LABELS[protocol], fontsize=8, x=0.02, ha="left")
    return fig


def heatmap_error_figure(scores: pd.DataFrame, protocol: str) -> plt.Figure:
    """The error version of `heatmap_figure`. Left: each cell's mean squared error in theta*
    (targets pooled over both semesters), on a sequential scale from 0. Right: that error divided by
    the within-domain forecast's error in the same column, coloured on a log scale centred on 1, so
    2 (twice the within-domain error) and 0.5 (half of it) sit equally far from white."""
    mse = (scores[scores.protocol == protocol].pivot_table(index="source", columns="scored", values="mse")
           .reindex(index=DOMAINS, columns=DOMAINS).to_numpy(float))
    ratio = mse / np.diag(mse)[None, :]
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.0), layout="constrained")
    im = axes[0].imshow(mse, cmap="Reds", vmin=0, vmax=float(np.nanmax(mse)))
    reach = max(1.0, float(np.nanmax(np.abs(np.log2(ratio)))))
    im2 = axes[1].imshow(np.log2(ratio), cmap="RdBu_r", vmin=-reach, vmax=reach)
    for ax, values, colour_of, fmt in ((axes[0], mse, lambda v: v / np.nanmax(mse), "{:.2f}"),
                                       (axes[1], ratio, lambda v: abs(np.log2(v)) / reach, "{:.1f}")):
        for i in range(4):
            for j in range(4):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if colour_of(values[i, j]) > 0.6 else "black")
        ax.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=90, fontsize=6)
        ax.set_yticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS] if ax is axes[0] else [], fontsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    axes[0].set_title("mean squared error", fontsize=7, color="grey", loc="left")
    axes[1].set_title("÷ within-domain error", fontsize=7, color="grey", loc="left")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    bar = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    ticks = [t for t in range(-int(np.floor(reach)), int(np.floor(reach)) + 1) if t % 2 == 0]  # every fourfold
    bar.set_ticks(ticks, labels=[f"{2.0 ** t:g}" for t in ticks])  # the log2 colour scale, labelled as ratios
    axes[0].set_ylabel("Transcripts from", fontsize=7)
    fig.supxlabel("Calibrated and scored on", fontsize=7)
    fig.suptitle(LABELS[protocol], fontsize=8, x=0.02, ha="left")
    return fig


def transfer_figure(transferred: pd.DataFrame, own: pd.DataFrame, tests: pd.DataFrame) -> plt.Figure:
    """Per protocol, the prediction score of maths and science transcripts scored on coding and
    cyber theta* (solid) beside the same models' score from their own coding and cyber transcripts
    (pale): a large circle at the mean over models, a box over its 95% t-interval and the
    per-model scores as small points. Stars from `tests` (protocol, p): the transferred forecasts'
    log likelihood ratio against the references' coding/cyber distribution, a target model pooled
    over its cells (likelihood_ratio.pooled_tests)."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    rng = np.random.default_rng(0)
    values = {(p, k): d[d.protocol == p].score.to_numpy(float)
              for p in protocols for k, d in (("transfer", transferred), ("own", own))}
    bottom, top = gen.padded_axis(np.concatenate([v for v in values.values() if len(v)]))
    fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    for i, p in enumerate(protocols):
        for j, (kind, alpha) in enumerate([("transfer", 1.0), ("own", 0.35)]):
            v = values[(p, kind)]
            if not len(v):
                continue
            x, mean, half = i + (j - 0.5) * 0.36, float(v.mean()), ci95(v)
            if np.isfinite(half):
                ax.add_patch(plt.Rectangle((x - 0.13, max(mean - half, bottom)), 0.26,
                                           min(mean + half, top) - max(mean - half, bottom),
                                           facecolor=COLORS[p], edgecolor="none", alpha=0.35 * alpha, zorder=2))
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=8, color=COLORS[p], alpha=0.5 * alpha + 0.2,
                       linewidths=0, edgecolors="none", zorder=3, clip_on=False)
            ax.scatter([x], [mean], s=28, color=COLORS[p], alpha=alpha, linewidths=0, edgecolors="none", zorder=5)
            if kind == "transfer" and p in set(tests.protocol):
                star = stars(float(tests.set_index("protocol").loc[p, "p"]))
                if star:
                    ax.text(x, min(mean + (half if np.isfinite(half) else 0), top), star, ha="center", va="bottom",
                            fontsize=7, color=COLORS[p], zorder=6)
    ax.set_xticks(range(len(protocols)), [LABELS[p] for p in protocols])
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(bottom, top)
    ax.set_ylabel(gen.SCORE_LABEL)
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker="o", ls="none", ms=5, alpha=a, label=l)
                        for a, l in [(1.0, "Maths/science transcripts"), (0.35, "Own coding/cyber transcripts")]],
               loc="outside lower center", ncol=1, frameon=False)
    return fig
