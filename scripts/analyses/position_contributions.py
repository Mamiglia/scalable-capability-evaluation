"""Where in the transcript the capability signal sits: pairwise judging on character thirds, one panel per domain.

Every rendered transcript (reasoning shown) is cut into three equal character spans and the judge sees one subset of
them, the rest replaced by "..."; the runs are the teammate's thirds ablation (src/challenges/thirds on main, logs
published to LASR-G5/judging). Prediction score = -log2(summed squared error / summed squared error of easy-item
prediction) over a domain's target models, in bits: 0 is easy-item prediction, +1 is half its error, -1 twice it.
On one axis per domain, above zero: the score with only that third shown (singular contribution); below zero: the
reduction in score when that third is removed and the other two are shown (unique contribution), drawn downward.
Dashed line: the full transcript.
Only the removal runs are read (RUNS). Each condition is scored on the (bench, item, left, right) rows judged in all
four conditions of its cell, since a third too short to judge drops that pair from that run only. Judges: the best
reference judge of each cell (RUNS), its full-transcript run as "full".
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
# the removal runs only (the paper reports no third kept alone), so kept-alone runs do not shrink the common pairs
REMOVED = ("12", "13", "23")
WHOLE_REMOVED = {k: WHOLE_TRAIN[k] for k in REMOVED}
FORTY_REMOVED = {k: FORTY_ITEMS[k] for k in REMOVED}
# domain -> semester -> (judge, full-transcript hash, thirds hashes): the best judge on references (lowest leave-one-out
# error, linear conversion) of each cell; Science 2026-I MiniMax-M2 was judged on thirds with 40 train items only
RUNS = {"coding": {"2025-II": ("minimax-m1", "24d6cc32", WHOLE_REMOVED), "2026-I": ("kimi-k2-thinking", "24d6cc32", WHOLE_REMOVED)},
        "cyber": {"2025-II": ("deepseek-r1-0528", "24d6cc32", WHOLE_REMOVED), "2026-I": ("kimi-k2-thinking", "24d6cc32", WHOLE_REMOVED)},
        "math": {"2025-II": ("deepseek-r1-0528", "24d6cc32", WHOLE_REMOVED), "2026-I": ("kimi-k2-thinking", "24d6cc32", WHOLE_REMOVED)},
        "science": {"2025-II": ("minimax-m1", "24d6cc32", WHOLE_REMOVED), "2026-I": ("minimax-m2", "66c170cd", FORTY_REMOVED)}}
KEY = ["bench", "item", "model_left", "model_right"]
DRAWS, SEED = 2000, 0

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def runs_for(domain, semester, keep):
    """Condition -> verdicts on the `keep` items, every condition cut to the rows judged in all of them."""
    judge, full, hashes = RUNS[domain][semester]
    dirs = {"full": full} | hashes
    runs = {}
    for name, digest in dirs.items():
        path = Path(f"logs/judge/{domain}/{semester}/{judge}/{digest}")
        if not any(path.glob("*.eval")):
            raise FileNotFoundError(f"{domain} {semester} {name}: no logs in {path}")
        verdicts = from_logs(path)
        verdicts = verdicts[verdicts.item.astype(str).isin(keep)]
        runs[name] = verdicts.sort_values("imported", kind="stable").drop_duplicates(KEY, keep="first")
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
        keep = contributions.removal_items(task)
        runs, common = runs_for(domain, semester, keep)
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
        if alone in point:  # the kept-alone runs are not read (REMOVED)
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
    # the manuscript figure, as change in squared error from the full transcript: each third removed
    contributions.save_figures(
        errors, POSITIONS, DOMAIN_COLORS, OUT, "position_contributions", final="transcript_thirds",
        measures=("without",), baseline=True,
        caption="Pairwise judge on transcript thirds. Every rendered transcript (reasoning shown) is cut into three "
                "equal character spans and the judge sees the other two with one third replaced by \"...\". Judges, "
                "the best reference judge of each cell: Coding 2025-II MiniMax-M1 and 2026-I Kimi-K2-Thinking, Cyber "
                "2025-II DeepSeek-R1 and 2026-I Kimi-K2-Thinking, Maths 2025-II DeepSeek-R1 and 2026-I "
                "Kimi-K2-Thinking, Science 2025-II MiniMax-M1 and 2026-I MiniMax-M2, each on the same 40 train items "
                "(10 per difficulty quartile). Every condition is scored on the pairs judged in all four conditions of its cell. Grey point: "
                "easy-item prediction's mean change in squared error from the full transcript, for scale (untested).")
