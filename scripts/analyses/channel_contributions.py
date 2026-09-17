"""Which transcript channels carry the capability signal: pairwise judging (linear conversion), one panel per domain.

Prediction score = 1 - summed squared error / summed squared error of easy-item prediction over a domain's target
models. On one axis per domain, above zero: the prediction score with only that channel shown (singular
contribution); below zero: the reduction in prediction score when that channel is removed from the full transcript
(unique contribution), drawn downward. Dashed line: the full transcript. Every run is cut to the frozen 10 train items
per quartile of its semester (40 items).
  coding, cyber (2026-I, 8 targets): Kimi-K2-Thinking channel runs (configs/judges/channels/); full transcript =
    24d6cc32 on the same items. 2025-II was skipped (the judges' providers throttled it to about a day of collection).
  math, science (2025-II and 2026-I pooled, 16 targets): no tool calls, so the only comparison is reasoning shown vs reasoning hidden (= without CoT) and CoT only
    (configs/judges/channels/<judge>-only-reasoning.yml), with each semester's best pairwise judge with reasoning shown
    by reference leave-one-out MSE. The hidden run is also "only messages"; it is not drawn twice.
Dot = estimate, shaded box = 95% percentile interval over 2,000 bootstrap resamples of the targets (resampled within
semester; paired for reductions). Marks: paired t-tests of per-target squared error, uncorrected
(*** p<.001, ** p<.01, * p<.05, n.s.): above a singular estimate, the channel alone vs easy-item prediction; below a
unique estimate, the transcript without the channel vs the full transcript. The boxes are bootstrap intervals because
the prediction score is a ratio of summed errors, not a mean over targets, so it has no t interval.
channel_shares: one pie per domain, the share of characters the models themselves wrote (role prefixes removed) in
each channel, over the correct first-attempt transcripts of the references and targets on the same items and
semesters, read from SMALL_JUDGE_DATA (default /Volumes/T7/small-judge-remote/data).
Run from the repo root: uv run python scripts/analyses/channel_contributions.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from analysis import judges, loading
from analysis.challenges.signal_glm import DOMAIN_COLORS
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from evals.transcripts import read_transcripts
from judge.dataset import response_chars
from protocols.pairwise.judge import JudgeBT, from_logs, stratified_items

OUT = Path("analysis/results/figures/channels")
DATA = Path(os.environ.get("SMALL_JUDGE_DATA", "/Volumes/T7/small-judge-remote/data"))
# channel label -> (only-run, removed-run)
CHANNELS = {"CoT": ("only-reasoning", "drop-reasoning"), "Messages": ("only-text", "drop-text"),
            "Tool\ncalls": ("only-calls", "drop-calls"), "Tool\noutputs": ("only-outputs", "drop-outputs"),
            "Tool calls\n+ outputs": ("only-tools", "drop-tools")}
# domain -> semester -> (judge, {run: log dir hash}); None = the channel runs of configs/judges/channels/
RUNS = {"coding": {"2026-I": None}, "cyber": {"2026-I": None},
        "math": {"2025-II": ("deepseek-r1-0528", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"}),
                 "2026-I": ("glm-4.7", {"full": "66c170cd", "drop-reasoning": "7ea04f2b"})},
        "science": {"2025-II": ("minimax-m1", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"}),
                    "2026-I": ("minimax-m2", {"full": "66c170cd", "drop-reasoning": "7ea04f2b"})}}
PIE_CHANNELS = {"CoT": "only-reasoning", "Messages": "only-text", "Tool calls": "only-calls", "Tool outputs": "only-outputs"}
PIE_COLORS = {"CoT": "#4a3aa7", "Messages": "#eda100", "Tool calls": "#1baf7a", "Tool outputs": "#e87ba4"}
DRAWS, SEED, WIDTH = 2000, 0, 0.5
# COLOR_BY: "domain" (signal_glm DOMAIN_COLORS, one hue per panel) or "channel" (the pie chart's hues, shared
# across panels; the combined tools column takes a darker shade of the tool-calls green)
COLOR_BY = os.environ.get("CHANNEL_FIGURE_COLOR_BY", "domain")
CHANNEL_COLORS = {"CoT": PIE_COLORS["CoT"], "Messages": PIE_COLORS["Messages"], "Tool\ncalls": PIE_COLORS["Tool calls"],
                  "Tool\noutputs": PIE_COLORS["Tool outputs"], "Tool calls\n+ outputs": "#0f7a55",
                  "Full\ntranscript": "0.35"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def runs_for(domain, semester, keep):
    """Run name -> verdicts on the semester's 40 items."""
    spec = RUNS[domain][semester]
    if spec is not None:
        judge, hashes = spec
        runs = {name: (v := from_logs(f"logs/judge/{domain}/{semester}/{judge}/{h}"))[v.item.isin(keep)] for name, h in hashes.items()}
        config = JudgeConfig.from_cli(f"configs/judges/channels/{judge}-only-reasoning.yml")  # CoT only, if collected
        run_dir = JudgeBT(config, DomainConfig.from_cli(f"configs/domain/{domain}.yml")).log_dir(semester)
        if run_dir.exists():
            runs["only-reasoning"] = from_logs(run_dir)
        return runs
    base = from_logs(f"logs/judge/{domain}/{semester}/kimi-k2-thinking/24d6cc32")
    runs = {"full": base[base.item.isin(keep)]}
    for run in (run for pair in CHANNELS.values() for run in pair):
        config = JudgeConfig.from_cli(f"configs/judges/channels/kimi-k2-thinking-{run}.yml")
        run_dir = JudgeBT(config, DomainConfig.from_cli(f"configs/domain/{domain}.yml")).log_dir(semester)
        if run_dir.exists():
            runs[run] = from_logs(run_dir)
    return runs


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def domain_stats(domain):
    """Per channel: singular (score alone) and unique (reduction without it) estimates, intervals and tests."""
    errors, cohorts = {}, {}  # (semester, run) -> squared error per target
    for semester in RUNS[domain]:
        (task,) = loading.tasks(loading.domain(domain), [semester])
        refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
        keep = stratified_items(task.references.items, 10)
        cohorts[semester] = (task, keep)
        for name, verdicts in runs_for(domain, semester, keep).items():
            theta = judges.pairwise_theta(verdicts, refs, targets, "linear", None).reindex(truth.index)
            errors[semester, name] = ((theta - truth) ** 2).to_numpy()
        errors[semester, "easy"] = ((refs.mean() - truth) ** 2).to_numpy()
    semesters = list(RUNS[domain])
    names = [n for s, n in errors if s == semesters[0] and all((t, n) in errors for t in semesters)]
    pooled = {n: np.concatenate([errors[s, n] for s in semesters]) for n in names}
    size = {s: len(errors[s, "easy"]) for s in semesters}

    def score(pick):
        sse = {n: np.concatenate([errors[s, n][pick[s]] for s in semesters]).sum() for n in names}
        return {n: 1 - sse[n] / sse["easy"] for n in names if n != "easy"}

    rng = np.random.default_rng(SEED)
    point = score({s: np.arange(size[s]) for s in semesters})
    draws = [score({s: rng.integers(0, size[s], size[s]) for s in semesters}) for _ in range(DRAWS)]
    rows = []
    for label, (only, removed) in CHANNELS.items():
        if only in point:
            rows.append({"channel": label, "measure": "singular", "run": only, "estimate": point[only],
                         "lo": np.percentile([d[only] for d in draws], 2.5), "hi": np.percentile([d[only] for d in draws], 97.5),
                         "p": ttest_rel(pooled[only], pooled["easy"]).pvalue})
        if removed in point:
            drops = [d["full"] - d[removed] for d in draws]
            rows.append({"channel": label, "measure": "unique", "run": removed, "estimate": point["full"] - point[removed],
                         "lo": np.percentile(drops, 2.5), "hi": np.percentile(drops, 97.5),
                         "p": ttest_rel(pooled[removed], pooled["full"]).pvalue})
    table = pd.DataFrame(rows).assign(domain=domain, targets=len(pooled["easy"]), semesters=", ".join(semesters))
    full = {"domain": domain, "full_prediction_score": point["full"],
            "lo": np.percentile([d["full"] for d in draws], 2.5), "hi": np.percentile([d["full"] for d in draws], 97.5)}
    return table, full, cohorts


def channel_shares(cohorts):
    """Characters the models wrote in each channel, over the references' and targets' correct first-attempt transcripts."""
    totals = dict.fromkeys(PIE_CHANNELS, 0)
    for task, keep in cohorts.values():
        models = pd.concat([task.references.models, task.test.models])
        for model in models.index:
            for column in (c for c in models.columns if c.startswith("path_")):
                path = models.loc[model, column]
                if not isinstance(path, str):
                    continue
                items = set(task.references.items.loc[task.references.items.bench == column[5:], "item"]) & keep
                for label, spec in PIE_CHANNELS.items():
                    found = read_transcripts(DATA / path, require_correct=True, include_reasoning=True, items=items, channels=spec)
                    totals[label] += sum(response_chars(texts[0].text) for texts in found.values())
    return totals


OUT.mkdir(parents=True, exist_ok=True)
stats, fulls, shares = [], [], {}
for domain in RUNS:
    table, full, cohorts = domain_stats(domain)
    stats.append(table)
    fulls.append(full)
    shares[domain] = channel_shares(cohorts)
stats, fulls = pd.concat(stats, ignore_index=True), pd.DataFrame(fulls).set_index("domain")
stats.assign(channel=stats.channel.str.replace("\n", " ")).to_csv(OUT / "channel_contributions.csv", index=False)
fulls.to_csv(OUT / "channel_contributions_full_transcript.csv")
print(stats.assign(channel=stats.channel.str.replace("\n", " ")).drop(columns="run").round(3).to_string(index=False))
print(fulls.round(3).to_string())

# the figure reads on a 0-100 scale; the csvs keep the raw fractions
stats[["estimate", "lo", "hi"]] *= 100
fulls[["full_prediction_score", "lo", "hi"]] *= 100

def draw(stats, fulls, name, color_by):
    """One panel per domain, widths by channel count; the full transcript first, then per channel the singular
    estimate above zero and the unique reduction below. `color_by`: "domain" (one hue per panel) or "channel"
    (the pie chart's hues, shared across panels). Values arrive on the 0-100 scale."""
    stats = stats.assign(channel=stats.channel.str.replace("\n", " "))
    channels = {c.replace("\n", " "): c for c in CHANNELS}
    colors = {k.replace("\n", " "): v for k, v in CHANNEL_COLORS.items()}
    domains = [d for d in RUNS if d in set(stats.domain)]
    groups = {d: ["Full transcript"] + [c for c in channels if c in set(stats.loc[stats.domain == d, "channel"])] for d in domains}
    # one y scale for every panel, deep enough for the largest reduction interval
    low = float(np.floor(min(0, -stats.loc[stats.measure == "unique", "hi"].max(),
                             stats.loc[stats.measure == "singular", "lo"].min()) / 50) * 50)
    ticks = np.arange(low, 101, 50)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.3, 1.8), sharey=True,
                             gridspec_kw={"width_ratios": [len(g) + 0.6 for g in groups.values()], "wspace": 0.12})
    for ax, domain in zip(np.atleast_1d(axes), domains):
        part, x = stats[stats.domain == domain], np.arange(len(groups[domain]))
        color = colors["Full transcript"] if color_by == "channel" else DOMAIN_COLORS[domain]
        full = fulls.loc[domain]
        ax.axhline(full.full_prediction_score, color="0.5", lw=0.8, ls="--", zorder=0)
        ax.axhline(0, color="black", lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((x[0] - WIDTH / 2, full.lo), WIDTH, full.hi - full.lo, color=color, alpha=0.3, linewidth=0, zorder=1))
        ax.plot(x[0], full.full_prediction_score, "o", ms=4, mfc=color, mec="black", mew=0.4, zorder=2)
        for position, label in zip(x[1:], groups[domain][1:]):
            if color_by == "channel":
                color = colors[label]
            for measure, sign, alpha, face in (("singular", 1, 0.3, color), ("unique", -1, 0.6, "white")):
                row = part[(part.channel == label) & (part.measure == measure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                lo, hi = sorted((sign * row.lo, sign * row.hi))
                ax.add_patch(plt.Rectangle((position - WIDTH / 2, lo), WIDTH, hi - lo, color=color, alpha=alpha, linewidth=0, zorder=1))
                ax.plot(position, sign * row.estimate, "o", ms=4, mfc=face, mec="black", mew=0.4, zorder=2)
                # singular marks in one row above the panel, clear of the full-transcript line; unique marks under their box
                ax.text(position, 106 if sign > 0 else lo - 4, stars(row.p), ha="center",
                        va="bottom" if sign > 0 else "top", fontsize=6.5)
        ax.set_ylim(low - 20, 122)
        ax.set_yticks(ticks, [f"{abs(v):g}" for v in ticks])  # below zero the axis reads as a reduction
        ax.spines["left"].set_bounds(low, 100)
        ax.spines["bottom"].set_visible(False)
        if ax is not np.atleast_1d(axes)[0]:  # one shared scale: the ticks are drawn once, on the left
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xticks(x, groups[domain], rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.set_title(domain.capitalize(), loc="left", fontsize=7, color="0.4", pad=6)
    np.atleast_1d(axes)[0].set_ylabel("Reduction (%)        Prediction score (%)", labelpad=2)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="0.6", mec="black", mew=0.4),
               plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec="black", mew=0.4),
               plt.Line2D([], [], color="0.5", lw=0.8, ls="--")]
    fig.legend(handles, ["Only this channel (above 0)", "Without this channel (below 0)", "Full transcript"], frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.98), handlelength=1.2, columnspacing=1.2, borderaxespad=0)
    for suffix in (".png", ".svg"):
        fig.savefig(OUT / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"{name}.png")


draw(stats, fulls, "channel_contributions", COLOR_BY)
draw(stats, fulls, "channel_contributions_by_domain" if COLOR_BY == "channel" else "channel_contributions_by_channel",
     "domain" if COLOR_BY == "channel" else "channel")
# the embedding protocol under the same conditions (scratchpad embedding_channels.py writes these tables)
embedding = OUT / "embedding_channel_contributions.csv"
if embedding.exists():
    e_stats = pd.read_csv(embedding)
    e_fulls = pd.read_csv(OUT / "embedding_channel_contributions_full_transcript.csv").set_index("domain")
    e_stats[["estimate", "lo", "hi"]] *= 100
    e_fulls[["full_prediction_score", "lo", "hi"]] *= 100
    e_stats = e_stats.rename(columns={"wilcoxon_p": "p"})
    draw(e_stats, e_fulls, "embedding_channel_contributions", COLOR_BY)

frame = pd.DataFrame(shares).T
share = frame.div(frame.sum(axis=1), axis=0)
share.to_csv(OUT / "channel_shares.csv")
print((share * 100).round(1).to_string())
fig, axes = plt.subplots(1, len(share), figsize=(4.4, 1.5))
for ax, domain in zip(axes, share.index):
    values = share.loc[domain]
    ax.pie(values, colors=[PIE_COLORS[c] for c in values.index], startangle=90, counterclock=False,
           wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
           autopct=lambda pct: f"{pct:.0f}%" if pct >= 10 else "", pctdistance=0.6, textprops={"fontsize": 7, "color": "white"})
    ax.set_title(domain.capitalize(), fontsize=7, color="0.4", pad=2)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=PIE_COLORS[c]) for c in PIE_COLORS], list(PIE_COLORS), frameon=False,
           ncol=len(PIE_COLORS), loc="lower center", bbox_to_anchor=(0.5, -0.08), handlelength=1, columnspacing=1)
for suffix in (".png", ".svg"):
    fig.savefig(OUT / f"channel_shares{suffix}", dpi=300, bbox_inches="tight")
print(OUT)
