"""Where in the transcript the capability signal sits: pairwise judging on character thirds, one panel per domain.

Every rendered transcript (reasoning shown) is cut into three equal character spans and the judge sees one subset of
them, the rest replaced by "..."; the runs are the teammate's thirds ablation (src/challenges/thirds on main, logs
published to LASR-G5/judging). Prediction score = -log2(summed squared error / summed squared error of easy-item
prediction) over a domain's target models, in bits: 0 is easy-item prediction, +1 is half its error, -1 twice it.
On one axis per domain, above zero: the score with only that third shown (singular contribution); below zero: the
reduction in score when that third is removed and the other two are shown (unique contribution), drawn downward.
Dashed line: the full transcript.
Each condition is scored on the (bench, item, left, right) rows judged in all seven conditions of its cell, since a
third too short to judge drops that pair from that run only.
  coding (2025-II MiniMax-M1, 2026-I MiniMax-M2), cyber (2025-II DeepSeek-R1), math and science (2026-I
  Kimi-K2-Thinking, 40 items): the top reference judge of each cell, its full-transcript baseline as "full".
Dot = estimate, shaded box = 95% percentile interval over 2,000 bootstrap resamples of the targets (resampled within
semester; paired for reductions). 
Marks: one-sided log likelihood ratio tests (src/analysis/challenges/likelihood_ratio.py), uncorrected (*** p<.001,
** p<.01, * p<.05, n.s.): above a singular estimate, the third alone against the references' distribution; below a
unique estimate, the full transcript against the transcript without the third.
Run from the repo root: uv run python scripts/analyses/position_contributions.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import judges, loading
from analysis.challenges import contributions
from analysis.challenges import likelihood_ratio as lr
from analysis.challenges.signal_glm import DOMAIN_COLORS
from protocols.pairwise.judge import from_logs

OUT = Path("analysis/results/figures/positions")
# kept thirds -> log dir hash, on the full train set and on the 40-item runs (src/challenges/thirds/README.md)
WHOLE_TRAIN = {"1": "66f9615b", "2": "574946b9", "3": "e4d05a32", "12": "b8f67d26", "13": "9d38c63d", "23": "97b40582"}
FORTY_ITEMS = {"1": "8311027f", "2": "f2b0de83", "3": "04e3f7ec", "12": "e5f74d2a", "13": "4e2e8bc9", "23": "03c21282"}
# position label -> (kept alone, kept when it is removed)
POSITIONS = {"First third": ("1", "23"), "Middle third": ("2", "13"), "Last third": ("3", "12")}
# domain -> semester -> (judge, full-transcript hash, thirds hashes); cells whose six thirds runs are complete
RUNS = {"coding": {"2025-II": ("minimax-m1", "24d6cc32", WHOLE_TRAIN), "2026-I": ("minimax-m2", "24d6cc32", WHOLE_TRAIN)},
        "cyber": {"2025-II": ("deepseek-r1-0528", "24d6cc32", WHOLE_TRAIN)},
        "math": {"2026-I": ("kimi-k2-thinking", "66c170cd", FORTY_ITEMS)},
        "science": {"2026-I": ("kimi-k2-thinking", "66c170cd", FORTY_ITEMS)}}
ROWS = ["Full transcript", *POSITIONS]
KEY = ["bench", "item", "model_left", "model_right"]
DRAWS, SEED, WIDTH = 2000, 0, 0.5

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def runs_for(domain, semester):
    """Condition -> verdicts, every condition cut to the rows judged in all of them."""
    judge, full, hashes = RUNS[domain][semester]
    dirs = {"full": full} | hashes
    runs = {}
    for name, digest in dirs.items():
        path = Path(f"logs/judge/{domain}/{semester}/{judge}/{digest}")
        if not any(path.glob("*.eval")):
            raise FileNotFoundError(f"{domain} {semester} {name}: no logs in {path}")
        runs[name] = from_logs(path).sort_values("imported", kind="stable").drop_duplicates(KEY, keep="first")
    common = runs["full"][KEY].drop_duplicates()
    for verdicts in runs.values():
        common = common.merge(verdicts[KEY].drop_duplicates())
    return {name: verdicts.merge(common) for name, verdicts in runs.items()}, len(common)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def domain_stats(domain):
    """Per third: singular (score alone) and unique (reduction without it) estimates, intervals and tests."""
    errors, coverage, models, llrs = {}, [], {}, []
    for semester in RUNS[domain]:
        (task,) = loading.tasks(loading.domain(domain), [semester])
        refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
        runs, common = runs_for(domain, semester)
        coverage.append({"domain": domain, "semester": semester, "judge": RUNS[domain][semester][0],
                         "targets": len(truth), "common_pairs": common})
        for name, verdicts in runs.items():
            theta = judges.pairwise_theta(verdicts, refs, targets, "linear", None).reindex(truth.index)
            errors[semester, name] = ((theta - truth) ** 2).to_numpy()
            llrs.append(lr.task_llr(lr.judge_rule(verdicts, "pairwise", "linear"), refs, truth)
                        .pipe(lambda d: d[d["shape"] == lr.SHAPE]).assign(domain=domain, semester=semester, condition=name))
        errors[semester, "easy"] = ((refs.mean() - truth) ** 2).to_numpy()
        models[semester] = list(truth.index)
    semesters = list(RUNS[domain])
    names = [n for s, n in errors if s == semesters[0] and all((t, n) in errors for t in semesters)]
    llr = pd.concat(llrs, ignore_index=True).query("condition in @names")
    pooled = {n: np.concatenate([errors[s, n] for s in semesters]) for n in names}
    size = {s: len(errors[s, "easy"]) for s in semesters}

    def score(pick):
        sse = {n: np.concatenate([errors[s, n][pick[s]] for s in semesters]).sum() for n in names}
        return {n: -np.log2(sse[n] / sse["easy"]) for n in names if n != "easy"}

    rng = np.random.default_rng(SEED)
    point = score({s: np.arange(size[s]) for s in semesters})
    draws = [score({s: rng.integers(0, size[s], size[s]) for s in semesters}) for _ in range(DRAWS)]
    rows = []
    for label, (alone, without) in POSITIONS.items():
        only = [d[alone] for d in draws]
        rows.append({"position": label, "measure": "singular", "run": f"thirds-{alone}", "estimate": point[alone],
                     "lo": np.percentile(only, 2.5), "hi": np.percentile(only, 97.5),
                     "p": contributions.llr_p(llr, alone)})
        drops = [d["full"] - d[without] for d in draws]
        rows.append({"position": label, "measure": "unique", "run": f"thirds-{without}",
                     "estimate": point["full"] - point[without],
                     "lo": np.percentile(drops, 2.5), "hi": np.percentile(drops, 97.5),
                     "p": contributions.llr_p(llr, "full", against=without)})
    table = pd.DataFrame(rows).assign(domain=domain, targets=len(pooled["easy"]), semesters=", ".join(semesters))
    full = {"domain": domain, "full_prediction_score": point["full"],
            "lo": np.percentile([d["full"] for d in draws], 2.5), "hi": np.percentile([d["full"] for d in draws], 97.5)}
    long = pd.DataFrame([{"domain": domain, "semester": s, "condition": n, "model": m, "err": e}
                         for (s, n), errs in errors.items() for m, e in zip(models[s], errs)])
    return table, full, pd.DataFrame(coverage), long, llr


def draw(stats, fulls, name):
    """One panel per domain, the thirds as rows: to the right of zero the prediction score with only that third
    (scale on top), to the left the reduction when it is removed (scale on the bottom). Values are in bits."""
    domains = [d for d in RUNS if d in set(stats.domain)]
    left = float(np.floor(min(0, -stats.loc[stats.measure == "unique", "hi"].max())))
    right = float(np.ceil(max(stats.loc[stats.measure == "singular", "hi"].max(), fulls.hi.max())))
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2, 2.1), sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0.1})
    axes = np.atleast_1d(axes)
    for ax, domain in zip(axes, domains):
        part = stats[stats.domain == domain]
        color = DOMAIN_COLORS[domain]
        full = fulls.loc[domain]
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.92", lw=0.5)  # one rule per row, to carry the eye across panels
        ax.axvline(full.full_prediction_score, color="0.5", lw=0.8, ls="--", zorder=0)
        ax.axvline(0, color="black", lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((full.lo, -WIDTH / 2), full.hi - full.lo, WIDTH, color=color, alpha=0.3,
                                   linewidth=0, zorder=1))
        ax.plot(full.full_prediction_score, 0, "o", ms=4, mfc=color, mec="black", mew=0.4, zorder=2)
        for label in ROWS[1:]:
            position = ROWS.index(label)
            for measure, sign, alpha, face in (("singular", 1, 0.3, color), ("unique", -1, 0.6, "white")):
                row = part[(part.position == label) & (part.measure == measure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                lo, hi = sorted((sign * row.lo, sign * row.hi))
                ax.add_patch(plt.Rectangle((lo, position - WIDTH / 2), hi - lo, WIDTH, color=color, alpha=alpha,
                                           linewidth=0, zorder=1))
                ax.plot(sign * row.estimate, position, "o", ms=4, mfc=face, mec="black", mew=0.4, zorder=2)
                # one column of marks per side, pinned just inside the panel edges, rather than hugging each estimate
                ax.text(0.98 if sign > 0 else 0.02, position, stars(row.p), va="center",
                        ha="right" if sign > 0 else "left", fontsize=6.5, transform=ax.get_yaxis_transform())
        ax.set_xlim(left - 1.1, right + 1.1)
        ax.set_ylim(len(ROWS) - 0.4, -0.7)  # the full transcript on top, the thirds under it
        ax.set_xticks(np.arange(left, 0.01, 2), [f"{abs(v):g}" for v in np.arange(left, 0.01, 2)])
        ax.spines["bottom"].set_bounds(left, 0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        top = ax.secondary_xaxis("top")
        top.set_xticks(np.arange(0, right + 0.01, 2))
        top.spines["top"].set_bounds(0, right)
        top.tick_params(labelsize=7, width=0.8)
        ax.set_title(domain.capitalize(), loc="left", fontsize=7, color="0.4", pad=14)
    axes[0].set_yticks(range(len(ROWS)), ROWS)
    fig.supxlabel("Reduction (bits)", fontsize=8, y=-0.02)
    fig.text(0.5, 1.12, "Prediction score (bits)", ha="center", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="0.6", mec="black", mew=0.4),
               plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec="black", mew=0.4),
               plt.Line2D([], [], color="0.5", lw=0.8, ls="--")]
    fig.legend(handles, ["Only this third (right)", "Without this third (left)", "Full transcript"],
               frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.18), handlelength=1.2,
               columnspacing=1.2, borderaxespad=0)
    for suffix in (".png", ".svg"):
        fig.savefig(OUT / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"{name}.png")


if __name__ == "__main__":  # importable, so a side analysis can redraw these estimates without recomputing them
    OUT.mkdir(parents=True, exist_ok=True)
    stats, fulls, coverage, errors, llrs = [], [], [], [], []
    for domain in RUNS:
        table, full, seen, long, llr = domain_stats(domain)
        stats.append(table)
        fulls.append(full)
        coverage.append(seen)
        errors.append(long)
        llrs.append(llr)
    errors, llrs = pd.concat(errors, ignore_index=True), pd.concat(llrs, ignore_index=True)
    llrs.to_csv(OUT / "position_llr.csv", index=False)  # every target's log likelihood ratio per run and dropped reference
    errors.to_csv(OUT / "position_errors.csv", index=False)  # every target's squared error, for redrawing
    stats, fulls = pd.concat(stats, ignore_index=True), pd.DataFrame(fulls).set_index("domain")
    coverage = pd.concat(coverage, ignore_index=True)
    stats.to_csv(OUT / "position_contributions.csv", index=False)
    fulls.to_csv(OUT / "position_contributions_full_transcript.csv")
    coverage.to_csv(OUT / "position_coverage.csv", index=False)
    print(stats.drop(columns="run").round(3).to_string(index=False))
    print(fulls.round(3).to_string())
    print(coverage.to_string(index=False))
    draw(stats, fulls, "position_contributions")

    # the manuscript figures, as change in squared error from the full transcript: each third kept alone, then each third removed
    contributions.save_figures(
        errors, POSITIONS, DOMAIN_COLORS, OUT, "position_contributions", final="transcript_thirds",
        caption="Pairwise judge on transcript thirds. Every rendered transcript (reasoning shown) is cut into three "
                "equal character spans and the judge sees one third alone, or the other two with that third replaced "
                "by \"...\". Judges, the top reference judge of each cell: Coding 2025-II MiniMax-M1 and 2026-I "
                "MiniMax-M2, Cyber 2025-II DeepSeek-R1 (full train set), Maths and Science 2026-I Kimi-K2-Thinking "
                "(40 train items). Every condition is scored on the pairs judged in all seven conditions of its cell.")
