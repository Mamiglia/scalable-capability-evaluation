"""Closed-source targets (Figure 9): six closed Coding 2026-I targets and the eight open ones of the same task,
predicted from the same eight 2025-II references on the frozen train items, reasoning hidden for every model and
protocol. Each target's squared error minus its baseline error, and per protocol and group the one-sample t-test of
the targets' log likelihood ratios against the baseline."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import likelihood_ratio as lr
from analysis.likelihood_ratio import num, p_text
from analysis.plots import stars
from analysis.stats import BASELINE
from capability.evaluate import forecast
from capability.task import Task
from protocols import embeddings
from protocols.direct_judge.judge import from_logs as direct_logs
from protocols.direct_judge.scoring import calibrate
from protocols.embeddings import EmbeddingRegression
from protocols.pairwise import from_logs as pairwise_logs
from protocols.pairwise.linear import weighted_theta
from protocols.transcript_features.features import FEATURE_SETS
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS, TranscriptRegression
from protocols.transcript_features.reader import feature_table

ROWS = ["pairwise", "direct", "embedding", "glm_regex"]  # the figure's rows, top to bottom
VENDOR_COLOURS = {"Anthropic": "#eb6834", "OpenAI": "#1baf7a", "Google": "#2a78d6"}
GROUP_SHADES = {"closed": "#b3b3b3", "open": "#dcdcdc"}


def on_train(table: pd.DataFrame, task: Task) -> pd.DataFrame:
    """A judge table's rows on the task's frozen train items."""
    return table.assign(item=table.item.astype(str)).merge(task.references.items[["bench", "item"]].astype({"item": str}))


def target_errors(task: Task, targets: list[str], pairwise_log: str, direct_log: str, embedding_variant: str,
                  features_view: str) -> pd.DataFrame:
    """Per target and protocol (the baseline, the references' mean θ*, included): θ*, forecast and squared error.

    pairwise_log, direct_log: the judge run dirs, reasoning hidden. embedding_variant: the embedding view with the
        reasoning stripped ("answer"). features_view: the deterministic features' view without reasoning ("hidden").
    """
    refs, truth = task.references.theta, task.truth.reindex(targets)
    theta_hat = {"pairwise": weighted_theta(on_train(pairwise_logs(pairwise_log), task), refs, targets),
                 "direct": calibrate(on_train(direct_logs(direct_log), task), refs).reindex(targets),
                 BASELINE: pd.Series(refs.mean(), index=targets)}
    for key, protocol in (("embedding", EmbeddingRegression(embedding_variant)),
                          ("glm_regex", TranscriptRegression(variant="regex", reasoning=features_view))):
        theta_hat[key] = forecast(task, protocol).set_index("model").theta_hat.reindex(targets)
    return pd.concat([pd.DataFrame({"model": targets, "protocol": p, "theta": truth.to_numpy(), "theta_hat": t.to_numpy(),
                                    "error": ((t - truth) ** 2).to_numpy()}) for p, t in theta_hat.items()], ignore_index=True)


def target_llr(task: Task, targets: list[str], pairwise_log: str, direct_log: str, embedding_variant: str,
               features_view: str, shape: str) -> pd.DataFrame:
    """Per target and protocol, the log likelihood ratio against the baseline on the full reference set, in `shape`
    ("t" or "normal"). Columns: model, shape, llr, dropped ("-"), protocol."""
    refs, truth = task.references.theta, task.truth.reindex(targets)
    cohorts = (task.references, task.test)
    rules = {"pairwise": lr.judge_rule(on_train(pairwise_logs(pairwise_log), task), "pairwise", "linear"),
             "direct": lr.judge_rule(on_train(direct_logs(direct_log), task), "direct", "raw"),
             "embedding": lr.feature_rule(RidgeCV(alphas=embeddings.ALPHAS),
                                          pd.concat([EmbeddingRegression(embedding_variant).get_features(c) for c in cohorts])),
             "glm_regex": lr.feature_rule(make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS)),
                                          pd.concat([feature_table(c, reasoning=features_view)[FEATURE_SETS["regex"]]
                                                     for c in cohorts]))}
    return pd.concat([pd.DataFrame({"model": truth.index, "shape": shape, "llr": lr.log_lr(rule, refs, truth)[shape].to_numpy(),
                                    "dropped": "-", "protocol": name}) for name, rule in rules.items()], ignore_index=True)


def error_deltas(errors: pd.DataFrame) -> pd.DataFrame:
    """`errors` (target_errors, with `group`) with `delta`: each target's squared error minus its baseline error."""
    base = errors[errors.protocol == BASELINE].set_index(["group", "model"]).error
    return errors.assign(delta=errors.error - errors.set_index(["group", "model"]).index.map(base).to_numpy())


def group_tests(llr: pd.DataFrame) -> pd.DataFrame:
    """Per protocol and group: n, the mean log likelihood ratio, its standard error, the two-sided one-sample t-test p
    and the mark drawn: stars when the mean is positive and p < .05, else "n.s."."""
    rows = []
    for (protocol, group), g in llr.groupby(["protocol", "group"]):
        t = lr.pooled_tests(g).iloc[0]
        p = 2 * min(t.p, 1 - t.p)  # pooled_tests gives the upper tail
        rows.append(dict(protocol=protocol, group=group, n=t.n, llr=t.llr, se=t.se, p=p,
                         mark=stars(p, "n.s.") if t.llr > 0 else "n.s."))
    return pd.DataFrame(rows)


def frontier_figure(deltas: pd.DataFrame, tests: pd.DataFrame, vendors: dict[str, str], display: dict) -> plt.Figure:
    """One row per protocol, closed then open targets: each target's change in squared error from its baseline error
    (closed targets in their developer's colour), a white circle at the group's mean over a grey box of its 95%
    t-interval, and the group's mark at the right. vendors: closed target -> developer."""
    fig, ax = plt.subplots(figsize=(2.3, 3.0))
    rng = np.random.default_rng(0)
    marks = tests.set_index(["protocol", "group"]).mark
    yticks = []
    for i, protocol in enumerate(ROWS):
        for j, group in enumerate(("closed", "open")):
            y = -(i * 3 + j * 1.05)
            e = deltas[(deltas.protocol == protocol) & (deltas.group == group)]
            mean = e.delta.mean()
            half = scipy_stats.t.ppf(0.975, len(e) - 1) * e.delta.std(ddof=1) / np.sqrt(len(e))
            ax.add_patch(plt.Rectangle((mean - half, y - 0.3), 2 * half, 0.6, color=GROUP_SHADES[group], lw=0, zorder=1))
            colours = [VENDOR_COLOURS[vendors[m]] if group == "closed" else "#9a9a9a" for m in e.model]
            ax.scatter(e.delta, y + rng.uniform(-0.25, 0.25, len(e)), s=32 if group == "closed" else 22, c=colours,
                       lw=0, zorder=2, clip_on=False)
            ax.scatter([mean], [y], s=70, facecolor="white", edgecolor="#333333", lw=1.2, zorder=3, clip_on=False)
            mark = marks[(protocol, group)]
            ax.text(6.5, y, mark, va="center", ha="left", fontsize=7 if mark != "n.s." else 6,
                    color="#555555" if group == "closed" else "#a6a6a6", clip_on=False)
        yticks.append(-(i * 3 + 0.525))
    ax.axvline(0, color="#444444", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-7.5, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.spines["bottom"].set_bounds(-6, 6)
    ax.set_yticks(yticks, [display["protocol_names"][p].replace(" ", "\n", 1) for p in ROWS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-(3 * 3 + 1.05) - 0.6, 0.6)
    ax.set_xlabel("Δ squared error vs baseline")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_SHADES[g], lw=0) for g in ("closed", "open")]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=5.5, mew=0, color=c) for c in VENDOR_COLOURS.values()]
    handles.insert(2, plt.Line2D([], [], ls="", marker=""))  # blank: column 1 = closed/open, column 2 = developers
    fig.legend(handles, ["Closed models", "Open models", "", *VENDOR_COLOURS], loc="lower center",
               bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)
    return fig


def caption(deltas: pd.DataFrame, tests: pd.DataFrame, display: dict, pairwise_judge: str, direct_judge: str) -> str:
    """The figure's .txt: what it draws, every target's change in squared error, each group's mean and interval and
    its log likelihood ratio test."""
    names = display["protocol_names"]
    by = tests.set_index(["protocol", "group"])
    lines = []
    for protocol in ROWS:
        lines.append(f"{names[protocol]}:")
        for group in ("closed", "open"):
            e = deltas[(deltas.protocol == protocol) & (deltas.group == group)]
            mean = e.delta.mean()
            half = scipy_stats.t.ppf(0.975, len(e) - 1) * e.delta.std(ddof=1) / np.sqrt(len(e))
            t = by.loc[(protocol, group)]
            lines.append(f"  {group} ({len(e)} targets): mean change {num(mean, sign=True)} (95% CI {num(mean - half)} to "
                         f"{num(mean + half)}); mean log likelihood ratio {num(t.llr)} (SE {num(t.se)}), {p_text(t.p)}, "
                         f"marked {t.mark}; per target " + "; ".join(f"{r.model} {num(r.delta, sign=True)}"
                                                                       for r in e.itertuples()) + ".")
    return ("Closed-source targets: each target's squared error minus its baseline error (below 0: better than the "
            "baseline, the references' mean θ*), reasoning hidden for every model and protocol. Six closed Coding 2026-I "
            "targets (coloured by developer) and the eight open targets of the same task, both predicted from the same "
            "eight 2025-II references on the frozen train items. Pairwise judge: "
            f"{pairwise_judge}, reasoning hidden, linear conversion; Absolute scoring: {direct_judge}, reasoning hidden, "
            "raw ratings; Embedding: the answer vectors with the reasoning stripped; Deterministic features: the regex "
            "features of the text without reasoning; both regressions as in the main analysis. The white circle is a "
            "group's mean and the box its 95% t-interval. Marks: one-sample t-test of the targets' log likelihood ratios "
            "against the baseline's distribution (full reference set), two-sided, marked only when the mean is positive; "
            "* p < .05, ** p < .01, *** p < .001, n.s. not significant.\n\n" + "\n".join(lines) + "\n")
