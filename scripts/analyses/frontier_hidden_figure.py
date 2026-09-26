"""Closed-source targets (Figure 9 of docs/manuscript_draft.pdf): each target's change in squared error from its
baseline error, reasoning hidden for every model and protocol.

Six closed targets (coding_frontier 2026-I) and the eight open 2026-I targets (coding), both predicted from the same
eight 2025-II references on the 100 frozen train items:
  Pairwise judge    kimi-k2-thinking, reasoning hidden (condition 9c1d3abb), JudgeLinear (weighted_theta)
  Absolute scoring  deepseek-v3.2 (the Coding 2026-I leave-one-out pick), reasoning hidden, raw ratings
  Embeddings        Qwen3-Embedding-8B "answer" vectors (reasoning stripped), ridge as the main analysis
  Deterministic     13 regex features, "hidden" view (text only), ridge as the main analysis
Box: mean and its 95% t-interval over a group's targets. Marks: one-sample t-test of the targets' log likelihood
ratios against the baseline (src/analysis/challenges/likelihood_ratio.py), per protocol and group.

Writes analysis/results/frontier_hidden/{frontier_hidden_errors,frontier_hidden_llr}.csv and
analysis/results/figures/paper/frontier.{pdf,png,svg}. Run from the repo root:
    PYTHONPATH=src uv run python scripts/analyses/frontier_hidden_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from capability.evaluate import forecast  # noqa: E402
from capability.task import Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols import embeddings  # noqa: E402
from protocols.direct_judge.judge import from_logs as direct_logs  # noqa: E402
from protocols.direct_judge.scoring import calibrate  # noqa: E402
from protocols.embeddings import EmbeddingRegression  # noqa: E402
from protocols.pairwise import from_logs as pairwise_logs  # noqa: E402
from protocols.pairwise.linear import weighted_theta  # noqa: E402
from protocols.transcript_features.features import FEATURE_SETS  # noqa: E402
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS, TranscriptRegression  # noqa: E402
from protocols.transcript_features.reader import feature_table  # noqa: E402

OUT, FINAL = Path("analysis/results/frontier_hidden"), Path("analysis/results/figures/paper")
SEMESTER = "2026-I"
CLOSED = {"claude-opus-4.7": "Anthropic", "claude-opus-4.8": "Anthropic", "claude-fable-5": "Anthropic",
          "gpt-5": "OpenAI", "gpt-5.6-sol": "OpenAI", "gemini-3.5-flash": "Google"}
VENDOR = {"Anthropic": "#eb6834", "OpenAI": "#1baf7a", "Google": "#2a78d6"}
PAIRWISE = {"closed": "logs/judge/coding_frontier/2026-I/kimi-k2-thinking/9c1d3abb",
            "open": "logs/judge/coding/2026-I/kimi-k2-thinking/9c1d3abb"}
DIRECT = {"closed": "logs/direct_reasoning-hidden/coding_frontier/2026-I/deepseek-v3.2/9d52d41c",
          "open": "logs/direct_reasoning-hidden/coding/2026-I/deepseek-v3.2/9d52d41c"}
ROWS = ["Pairwise judge", "Absolute scoring", "Embeddings", "Deterministic features"]


def on_train(table: pd.DataFrame, task: Task) -> pd.DataFrame:
    return table.assign(item=table.item.astype(str)).merge(task.references.items[["bench", "item"]].astype({"item": str}))


def errors(group: str, task: Task, targets: list[str]) -> pd.DataFrame:
    """Each target's squared error per protocol, and the baseline's (the references' mean theta*)."""
    refs, truth = task.references.theta, task.truth.reindex(targets)
    theta_hat = {"Pairwise judge": weighted_theta(on_train(pairwise_logs(PAIRWISE[group]), task), refs, targets),
                 "Absolute scoring": calibrate(on_train(direct_logs(DIRECT[group]), task), refs).reindex(targets),
                 "Baseline": pd.Series(refs.mean(), index=targets)}
    for label, protocol in (("Embeddings", EmbeddingRegression("answer")),
                            ("Deterministic features", TranscriptRegression(variant="regex", reasoning="hidden"))):
        theta_hat[label] = forecast(task, protocol).set_index("model").theta_hat.reindex(targets)
    return pd.concat([pd.DataFrame({"group": group, "model": targets, "protocol": p, "theta": truth.to_numpy(),
                                    "theta_hat": t.to_numpy(), "error": ((t - truth) ** 2).to_numpy()})
                      for p, t in theta_hat.items()], ignore_index=True)


def llr_rows(group: str, task: Task, targets: list[str]) -> pd.DataFrame:
    """Each target's log likelihood ratio per protocol against the baseline (full reference set)."""
    refs, truth = task.references.theta, task.truth.reindex(targets)
    cohorts = (task.references, task.test)
    rules = {"Pairwise judge": lr.judge_rule(on_train(pairwise_logs(PAIRWISE[group]), task), "pairwise", "linear"),
             "Absolute scoring": lr.judge_rule(on_train(direct_logs(DIRECT[group]), task), "direct", "raw"),
             "Embeddings": lr.feature_rule(RidgeCV(alphas=embeddings.ALPHAS),
                                           pd.concat([EmbeddingRegression("answer").get_features(c) for c in cohorts])),
             "Deterministic features": lr.feature_rule(make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS)),
                                                       pd.concat([feature_table(c, reasoning="hidden")[FEATURE_SETS["regex"]] for c in cohorts]))}
    rows = pd.concat([lr.task_llr(rule, refs, truth).assign(protocol=name) for name, rule in rules.items()])
    return rows[(rows["shape"] == lr.SHAPE) & (rows.dropped == "-")].assign(group=group, domain="coding", semester=SEMESTER)


frontier = Task.from_semester(SEMESTER, DomainConfig.from_cli(Path("configs/domain/coding_frontier.yml")))
coding = Task.from_semester(SEMESTER, DomainConfig.from_cli(Path("configs/domain/coding.yml")))
assert frontier.references.theta.sort_index().equals(coding.references.theta.sort_index()), "reference sets differ"
groups = (("closed", frontier, list(CLOSED)), ("open", coding, list(coding.test.models.index)))
raw = pd.concat([errors(*g) for g in groups], ignore_index=True)
llr = pd.concat([llr_rows(*g) for g in groups], ignore_index=True)
OUT.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT / "frontier_hidden_errors.csv", index=False, float_format="%.4f")
llr.to_csv(OUT / "frontier_hidden_llr.csv", index=False, float_format="%.5f")

# each target's error minus its own baseline error: 0 = no better than the baseline
base = raw[raw.protocol == "Baseline"].set_index(["group", "model"]).error
delta = raw.assign(error=raw.error - raw.set_index(["group", "model"]).index.map(base).to_numpy())
p_value = {key: float(lr.pooled_tests(g).p.iloc[0]) for key, g in llr.groupby(["protocol", "group"])}  # mean LLR > 0
for (protocol, group), p in p_value.items():
    print(f"LLR {protocol:24s} {group:6s} mean {llr[(llr.protocol == protocol) & (llr.group == group)].llr.mean():+.2f}  p = {p:.4f}")
print(delta.groupby(["protocol", "group"]).error.agg(["median", "mean", "count"]).unstack("group").round(2).to_string())

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(2.3, 3.0))
rng = np.random.default_rng(0)
yticks = []
for i, protocol in enumerate(ROWS):
    for j, (group, shade) in enumerate((("closed", "#b3b3b3"), ("open", "#dcdcdc"))):
        y = -(i * 3 + j * 1.05)
        e = delta[(delta.protocol == protocol) & (delta.group == group)]
        mean = e.error.mean()
        half = stats.t.ppf(0.975, len(e) - 1) * e.error.std(ddof=1) / np.sqrt(len(e))
        ax.add_patch(plt.Rectangle((mean - half, y - 0.3), 2 * half, 0.6, color=shade, lw=0, zorder=1))
        colours = [VENDOR[CLOSED[m]] if group == "closed" else "#9a9a9a" for m in e.model]
        ax.scatter(e.error, y + rng.uniform(-0.25, 0.25, len(e)), s=32 if group == "closed" else 22, c=colours,
                   lw=0, zorder=2, clip_on=False)
        ax.scatter([mean], [y], s=70, facecolor="white", edgecolor="#333333", lw=1.2, zorder=3, clip_on=False)
        p = p_value[(protocol, group)]
        mark = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(6.5, y, mark, va="center", ha="left", fontsize=7 if mark != "n.s." else 6,
                color="#555555" if group == "closed" else "#a6a6a6", clip_on=False)
    yticks.append(-(i * 3 + 0.525))
ax.axvline(0, color="#444444", lw=0.8, ls="--", zorder=0)
ax.set_xlim(-7.5, 6)
ax.set_xticks([-6, -3, 0, 3, 6])
ax.spines["bottom"].set_bounds(-6, 6)
ax.set_yticks(yticks, [r.replace(" ", "\n", 1) if r != "Embeddings" else r for r in ROWS])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_ylim(-(3 * 3 + 1.05) - 0.6, 0.6)
ax.set_xlabel("Δ squared error vs baseline")
handles = [plt.Rectangle((0, 0), 1, 1, color="#b3b3b3", lw=0), plt.Rectangle((0, 0), 1, 1, color="#dcdcdc", lw=0)]
handles += [plt.Line2D([], [], marker="o", ls="", ms=5.5, mew=0, color=c) for c in VENDOR.values()]
handles.insert(2, plt.Line2D([], [], ls="", marker=""))  # blank: column 1 = closed/open, column 2 = labs
fig.legend(handles, ["Closed models", "Open models", "", *VENDOR], loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=2,
           frameon=False, handlelength=1.0, columnspacing=0.8)
for ext in ("pdf", "png", "svg"):
    fig.savefig(FINAL / f"frontier.{ext}", dpi=300, bbox_inches="tight")
print("wrote", FINAL / "frontier.pdf")
