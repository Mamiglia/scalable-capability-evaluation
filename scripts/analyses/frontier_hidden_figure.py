"""The frontier figure with reasoning hidden for every model and protocol (Figure 9 of docs/manuscript_draft.pdf).

Closed targets (coding_frontier 2026-I) vs the eight open 2026-I targets (coding), both predicted from the same eight
2025-II references. Inputs, all from generality-labs (transcripts, pairwise logs, published features/embeddings) except
where computed here from those transcripts:
  Pairwise judge    kimi-k2-thinking, reasoning-hidden condition 9c1d3abb (40 items), JudgeLinear (weighted_theta)
  Absolute scoring  deepseek-v3.2 (leave-one-out pick once coding's split direct runs are merged), reasoning hidden
  Embeddings        Qwen3-Embedding-8B "answer" vectors (reasoning stripped), ridge as the main analysis
  Deterministic     13 regex features, "hidden" view (text only), ridge as the main analysis
Squared error per target; baseline = the references' mean theta*.

Env: SMALL_JUDGE_DATA (transcripts), SMALL_JUDGE_EMBEDDINGS (npz root), FEATURES_ROOT (feature CSV root), OUT.
Modes: LLR=1 (log likelihood ratio axis), DELTA=1 (change from the baseline), RATIO=1; default squared error.
Run from the repo root: FEATURES_ROOT=... OUT=... LLR=1 uv run python scripts/analyses/frontier_hidden_figure.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from protocols.embeddings import EmbeddingRegression  # noqa: E402
from capability.evaluate import forecast  # noqa: E402
from capability.task import Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols.direct_judge.judge import from_logs as direct_logs  # noqa: E402
from protocols.direct_judge.scoring import calibrate  # noqa: E402
from protocols.pairwise import from_logs as pairwise_logs  # noqa: E402
from protocols.pairwise.linear import weighted_theta  # noqa: E402
from protocols.transcript_features.protocol import TranscriptRegression  # noqa: E402
from reference import tables  # noqa: E402

tables.FEATURES = Path(os.environ["FEATURES_ROOT"])
OUT = Path(os.environ["OUT"])
SEMESTER = "2026-I"
CLOSED = {"claude-opus-4.7": "Anthropic", "claude-opus-4.8": "Anthropic", "claude-fable-5": "Anthropic",
          "gpt-5": "OpenAI", "gpt-5.6-sol": "OpenAI", "gemini-3.5-flash": "Google"}
VENDOR = {"Anthropic": "#eb6834", "OpenAI": "#1baf7a", "Google": "#2a78d6"}  # as the paper figure: orange, green, blue
PAIRWISE = {"closed": "logs/judge/coding_frontier/2026-I/kimi-k2-thinking/9c1d3abb",
            "open": "logs/judge/coding/2026-I/kimi-k2-thinking/9c1d3abb"}
DIRECT = {"closed": "logs/direct_reasoning-hidden/coding_frontier/2026-I/deepseek-v3.2/9d52d41c",
          "open": "logs/direct_reasoning-hidden/coding/2026-I/deepseek-v3.2/9d52d41c"}
ROWS = ["Pairwise judge", "Absolute scoring", "Embeddings", "Deterministic features"]


def on_train(table: pd.DataFrame, task: Task) -> pd.DataFrame:
    return table.assign(item=table.item.astype(str)).merge(task.references.items[["bench", "item"]].astype({"item": str}))


def predictions(group: str, task: Task, targets: list[str]) -> dict[str, pd.Series]:
    refs = task.references.theta
    out = {"Pairwise judge": weighted_theta(on_train(pairwise_logs(PAIRWISE[group]), task), refs, targets),
           "Absolute scoring": calibrate(on_train(direct_logs(DIRECT[group]), task), refs).reindex(targets)}
    for label, protocol in (("Embeddings", EmbeddingRegression("answer")),
                            ("Deterministic features", TranscriptRegression(variant="regex", reasoning="hidden"))):
        out[label] = forecast(task, protocol).set_index("model").theta_hat.reindex(targets)
    out["Baseline"] = pd.Series(refs.mean(), index=targets)
    return out


frontier = Task.from_semester(SEMESTER, DomainConfig.from_cli(Path("configs/domain/coding_frontier.yml")))
coding = Task.from_semester(SEMESTER, DomainConfig.from_cli(Path("configs/domain/coding.yml")))
assert frontier.references.theta.sort_index().equals(coding.references.theta.sort_index()), "reference sets differ"
open_targets = list(coding.test.models.index)
rows = []
for group, task, targets in (("closed", frontier, list(CLOSED)), ("open", coding, open_targets)):
    for protocol, theta_hat in predictions(group, task, targets).items():
        truth = task.truth.reindex(targets)
        rows += [dict(group=group, model=m, protocol=protocol, theta=truth[m], theta_hat=theta_hat[m],
                      error=(theta_hat[m] - truth[m]) ** 2) for m in targets]
errors = pd.DataFrame(rows)

from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from protocols import embeddings as emb  # noqa: E402
from analysis.challenges import likelihood_ratio as lr  # noqa: E402
from protocols.transcript_features.features import FEATURE_SETS  # noqa: E402
from protocols.transcript_features.protocol import ALPHAS as FEATURE_ALPHAS  # noqa: E402
from protocols.transcript_features.reader import feature_table  # noqa: E402


def llr_rows(group: str, task: Task, targets: list[str]) -> pd.DataFrame:
    refs, truth = task.references.theta, task.truth.reindex(targets)
    cohorts = (task.references, task.test)
    rules = {"Pairwise judge": lr.judge_rule(on_train(pairwise_logs(PAIRWISE[group]), task), "pairwise", "linear"),
             "Absolute scoring": lr.judge_rule(on_train(direct_logs(DIRECT[group]), task), "direct", "raw"),
             "Embeddings": lr.feature_rule(RidgeCV(alphas=emb.ALPHAS),
                                           pd.concat([EmbeddingRegression("answer").get_features(c) for c in cohorts])),
             "Deterministic features": lr.feature_rule(make_pipeline(StandardScaler(), RidgeCV(alphas=FEATURE_ALPHAS)),
                                                       pd.concat([feature_table(c, reasoning="hidden")[FEATURE_SETS["regex"]] for c in cohorts]))}
    out = [lr.task_llr(rule, refs, truth).assign(protocol=name) for name, rule in rules.items()]
    rows = pd.concat(out)
    return rows[(rows["shape"] == lr.SHAPE) & (rows.dropped == "-")].assign(group=group, domain="coding", semester=SEMESTER)


llr = pd.concat([llr_rows("closed", frontier, list(CLOSED)), llr_rows("open", coding, open_targets)])
llr.to_csv(OUT / "frontier_hidden_llr.csv", index=False, float_format="%.5f") if OUT.mkdir(parents=True, exist_ok=True) is None else None
LLR_P = {}
for (protocol, group), g in llr.groupby(["protocol", "group"]):
    t = lr.pooled_tests(g)
    p1 = float(t.p.iloc[0])
    LLR_P[(protocol, group)] = p1  # one-sided: does the protocol beat the baseline (mean LLR > 0)
    print(f"LLR {protocol:24s} {group:6s} mean {g.llr.mean():+.2f}  p = {LLR_P[(protocol, group)]:.4f}")
RATIO = os.environ.get("RATIO") == "1"  # each target's error over its own baseline error: 1 = no better than the baseline
DELTA = os.environ.get("DELTA") == "1"  # each target's error minus its own baseline error: 0 = no better than the baseline
raw = errors.copy()
if DELTA:
    base_err = errors[errors.protocol == "Baseline"].set_index(["group", "model"]).error
    errors = errors.assign(error=errors.error - errors.set_index(["group", "model"]).index.map(base_err).to_numpy())
if RATIO:
    base_err = errors[errors.protocol == "Baseline"].set_index(["group", "model"]).error
    errors = errors.assign(error=errors.error / errors.set_index(["group", "model"]).index.map(base_err).to_numpy())
OUT.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT / "frontier_hidden_errors.csv", index=False, float_format="%.4f")
summary = errors.groupby(["protocol", "group"]).error.agg(["median", "mean", "count"]).unstack("group")
print(summary.round(2).to_string())

LLRMODE = os.environ.get("LLR") == "1"
if LLRMODE:
    errors = llr.rename(columns={"llr": "error"})[["group", "model", "protocol", "error"]]
    DELTA = True  # linear axis, reference line at 0, marks drawn

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
        e = errors[(errors.protocol == protocol) & (errors.group == group)]
        from scipy import stats as st  # box = mean and its 95% t-interval over the group's targets
        med = e.error.mean()
        half = st.t.ppf(0.975, len(e) - 1) * e.error.std(ddof=1) / np.sqrt(len(e))
        q1, q3 = med - half, med + half
        ax.add_patch(plt.Rectangle((q1, y - 0.3), q3 - q1, 0.6, color=shade, lw=0, zorder=1))
        colours = [VENDOR[CLOSED[m]] if group == "closed" else "#9a9a9a" for m in e.model]
        ax.scatter(e.error, y + rng.uniform(-0.25, 0.25, len(e)), s=32 if group == "closed" else 22, c=colours,
                   lw=0, zorder=2, clip_on=False)
        ax.scatter([med], [y], s=70, facecolor="white", edgecolor="#333333", lw=1.2, zorder=3, clip_on=False)
        if DELTA:  # one-sample two-sided t-test of the change from the baseline, per group; mark coloured by group
            pval = LLR_P[(protocol, group)]  # the paper's test: log likelihood ratio against the baseline
            mark = "***" if pval < .001 else "**" if pval < .01 else "*" if pval < .05 else "n.s."
            ax.text(6.4 if LLRMODE else 6.5, y, mark, va="center", ha="left", fontsize=7 if mark != "n.s." else 6,
                    color="#555555" if group == "closed" else "#a6a6a6", clip_on=False)
            print("mark", protocol, group, mark)
    yticks.append(-(i * 3 + 0.525))
base = 0.0 if DELTA else 1.0 if RATIO else errors[errors.protocol == "Baseline"].error.median()
ax.axvline(base, color="#444444", lw=0.8, ls="--", zorder=0)
if not LLRMODE:
    ax.text(base + 0.15 if DELTA else base * 1.1, 0.55, "baseline", rotation=90, va="top", ha="left", fontsize=7, color="#444444")
if LLRMODE:
    ax.set_xlim(-6, 6)
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.spines["bottom"].set_bounds(-6, 6)
elif DELTA:
    ax.set_xlim(-7.5, 6)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.spines["bottom"].set_bounds(-6, 6)
else:
    ax.set_xscale("log")
    lim = (1e-3, 1e1) if RATIO else (1e-2, 1e1)
    ax.set_xlim(*lim)
    ax.set_xticks([t for t in (1e-3, 1e-2, 1e-1, 1, 1e1) if lim[0] <= t <= lim[1]])
    ax.spines["bottom"].set_bounds(*lim)
ax.set_yticks(yticks, [r.replace(" ", "\n", 1) if r != "Embeddings" else r for r in ROWS])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_ylim(-(3 * 3 + 1.05) - 0.6, 0.6)
ax.set_xlabel("Log likelihood ratio vs baseline" if LLRMODE else "Δ squared error vs baseline" if DELTA else "Squared error / baseline's" if RATIO else "Squared error")
handles = [plt.Rectangle((0, 0), 1, 1, color="#b3b3b3", lw=0), plt.Rectangle((0, 0), 1, 1, color="#dcdcdc", lw=0)]
handles += [plt.Line2D([], [], marker="o", ls="", ms=5.5, mew=0, color=c) for c in VENDOR.values()]
handles.insert(2, plt.Line2D([], [], ls="", marker=""))  # blank: column 1 = closed/open, column 2 = labs
fig.legend(handles, ["Closed models", "Open models", "", *VENDOR], loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=2,
           frameon=False, handlelength=1.0, columnspacing=0.8)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"frontier_hidden{'_llr' if LLRMODE else '_delta' if DELTA else '_ratio' if RATIO else ''}_ci.{ext}", dpi=300, bbox_inches="tight")
print("baseline median squared error", round(float(base), 2), "->", OUT / "frontier_hidden.pdf")
