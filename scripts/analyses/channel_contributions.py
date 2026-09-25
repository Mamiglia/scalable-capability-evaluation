"""Which transcript channels carry the capability signal: pairwise judging (linear conversion), one panel per domain.

Prediction score = -log2(summed squared error / summed squared error of easy-item prediction) over a domain's target
models, in bits: 0 is easy-item prediction, +1 is half its error, -1 twice it. On one axis per domain, above zero: the prediction score with only that channel shown (singular
contribution); below zero: the reduction in prediction score when that channel is removed from the full transcript
(unique contribution), drawn downward. Dashed line: the full transcript. Every run is cut to the frozen 10 train items
per quartile of its semester (40 items).
  coding, cyber (2025-II and 2026-I pooled, 16 targets): channel runs of each semester's best pairwise judge
    (configs/judges/channels/): Kimi-K2-Thinking on both in 2026-I, MiniMax-M1 coding and DeepSeek-R1-0528 cyber in
    2025-II; full transcript = 24d6cc32 on the same items.
  math, science (2025-II and 2026-I pooled, 16 targets): no tool calls, so the only comparison is reasoning shown vs reasoning hidden (= without CoT) and CoT only
    (configs/judges/channels/<judge>-only-reasoning.yml), with each semester's best pairwise judge with reasoning shown
    by reference leave-one-out MSE. The hidden run is also "only messages", and the CoT-only run also "without
    messages".
Dot = estimate, shaded box = 95% percentile interval over 2,000 bootstrap resamples of the targets (resampled within
semester; paired for reductions). 
Marks: one-sided log likelihood ratio tests (src/analysis/challenges/likelihood_ratio.py), uncorrected (*** p<.001,
** p<.01, * p<.05, n.s.): above a singular estimate, the channel alone against the references' distribution; below a
unique estimate, the full transcript against the transcript without the channel.
channel_shares: one pie per domain, the share of characters the models themselves wrote (role prefixes removed) in
each channel, over the correct first-attempt transcripts of the references and targets on the same items and
semesters, read from SCE_DATA (default data/).
Run from the repo root: uv run python scripts/analyses/channel_contributions.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import judges, loading
from analysis.challenges import contributions
from analysis.challenges import likelihood_ratio as lr
from plots import contributions as contributions_plots
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from evals.transcripts import read_transcripts
from judge.dataset import response_chars
from protocols.pairwise.judge import JudgeBT, from_logs, stratified_items

OUT = Path("analysis/results/figures/channels")
DATA = Path(os.environ.get("SCE_DATA", "data"))
# channel label -> (only-run, removed-run)
CHANNELS = {"CoT": ("only-reasoning", "drop-reasoning"), "Messages": ("only-text", "drop-text"),
            "Tool calls": ("only-calls", "drop-calls"), "Tool outputs": ("only-outputs", "drop-outputs"),
            "Tools": ("only-tools", "drop-tools")}
# domain -> semester -> (judge, {run: log dir hash}), or a judge name = its channel runs of configs/judges/channels/
RUNS = {"coding": {"2025-II": "minimax-m1", "2026-I": "kimi-k2-thinking"},
        "cyber": {"2025-II": "deepseek-r1-0528", "2026-I": "kimi-k2-thinking"},
        "math": {"2025-II": ("deepseek-r1-0528", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"}),
                 "2026-I": ("kimi-k2-thinking", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"})},
        "science": {"2025-II": ("minimax-m1", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"}),
                    "2026-I": ("minimax-m2", {"full": "24d6cc32", "drop-reasoning": "7ea04f2b"})}}
PIE_CHANNELS = {"CoT": "only-reasoning", "Messages": "only-text", "Tools": "only-tools"}
DRAWS, SEED = 2000, 0
# COLOR_BY: "domain" (signal_glm DOMAIN_COLORS, one hue per panel) or "channel" (the pie chart's hues, shared
# across panels; the combined tools column takes a darker shade of the tool-calls green)
COLOR_BY = os.environ.get("CHANNEL_FIGURE_COLOR_BY", "domain")

contributions_plots.rc()


def runs_for(domain, semester, keep):
    """Run name -> verdicts on the semester's 40 items."""
    spec = RUNS[domain][semester]
    if not isinstance(spec, str):
        judge, hashes = spec
        runs = {name: (v := from_logs(f"logs/judge/{domain}/{semester}/{judge}/{h}"))[v.item.isin(keep)] for name, h in hashes.items()}
        config = JudgeConfig.from_cli(f"configs/judges/channels/{judge}-only-reasoning.yml")  # CoT only, if collected
        run_dir = JudgeBT(config, DomainConfig.from_cli(f"configs/domain/{domain}.yml")).log_dir(semester)
        if run_dir.exists():
            # no tool calls: the reasoning alone is also the transcript without messages
            runs["only-reasoning"] = runs["drop-text"] = from_logs(run_dir)
        return runs
    base = from_logs(f"logs/judge/{domain}/{semester}/{spec}/24d6cc32")
    runs = {"full": base[base.item.isin(keep)]}
    for run in (run for pair in CHANNELS.values() for run in pair):
        config = JudgeConfig.from_cli(f"configs/judges/channels/{spec}-{run}.yml")
        run_dir = JudgeBT(config, DomainConfig.from_cli(f"configs/domain/{domain}.yml")).log_dir(semester)
        if run_dir.exists():
            runs[run] = from_logs(run_dir)
    return runs


def domain_stats(domain):
    """Per channel: singular (score alone) and unique (reduction without it) estimates, intervals and tests."""
    errors, cohorts, models, llrs = {}, {}, {}, []  # (semester, run) -> squared error per target; semester -> targets
    for semester in RUNS[domain]:
        (task,) = loading.tasks(loading.domain(domain), [semester])
        refs, targets, truth = task.references.theta, list(task.test.models.index), task.truth
        keep = stratified_items(task.references.items, 10)
        cohorts[semester] = (task, keep)
        for name, verdicts in runs_for(domain, semester, keep).items():
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
    for label, (only, removed) in CHANNELS.items():
        if only in point:
            alone = [d[only] for d in draws]
            rows.append({"channel": label, "measure": "singular", "run": only, "estimate": point[only],
                         "lo": np.percentile(alone, 2.5), "hi": np.percentile(alone, 97.5),
                         "p": contributions.llr_p(llr, only)})
        if removed in point:
            drops = [d["full"] - d[removed] for d in draws]
            rows.append({"channel": label, "measure": "unique", "run": removed, "estimate": point["full"] - point[removed],
                         "lo": np.percentile(drops, 2.5), "hi": np.percentile(drops, 97.5),
                         "p": contributions.llr_p(llr, "full", against=removed)})
    table = pd.DataFrame(rows).assign(domain=domain, targets=len(pooled["easy"]), semesters=", ".join(semesters))
    full = {"domain": domain, "full_prediction_score": point["full"],
            "lo": np.percentile([d["full"] for d in draws], 2.5), "hi": np.percentile([d["full"] for d in draws], 97.5)}
    long = pd.DataFrame([{"domain": domain, "semester": s, "condition": n, "model": m, "err": e}
                         for (s, n), errs in errors.items() for m, e in zip(models[s], errs)])
    return table, full, cohorts, long, llr


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
stats, fulls, shares, errors, llrs = [], [], {}, [], []
for domain in RUNS:
    table, full, cohorts, long, llr = domain_stats(domain)
    stats.append(table)
    fulls.append(full)
    errors.append(long)
    llrs.append(llr)
    shares[domain] = channel_shares(cohorts)
errors, llrs = pd.concat(errors, ignore_index=True), pd.concat(llrs, ignore_index=True)
llrs.to_csv(OUT / "channel_llr.csv", index=False)  # every target's log likelihood ratio per run and dropped reference
errors.to_csv(OUT / "channel_errors.csv", index=False)  # every target's squared error, so figures redraw without the logs
stats, fulls = pd.concat(stats, ignore_index=True), pd.DataFrame(fulls).set_index("domain")
stats.assign(channel=stats.channel.str.replace("\n", " ")).to_csv(OUT / "channel_contributions.csv", index=False)
fulls.to_csv(OUT / "channel_contributions_full_transcript.csv")
print(stats.assign(channel=stats.channel.str.replace("\n", " ")).drop(columns="run").round(3).to_string(index=False))
print(fulls.round(3).to_string())

contributions_plots.channel_figure(stats, fulls, RUNS, OUT, "channel_contributions", COLOR_BY)
contributions_plots.channel_figure(stats, fulls, RUNS, OUT, "channel_contributions_by_domain" if COLOR_BY == "channel"
                                   else "channel_contributions_by_channel", "domain" if COLOR_BY == "channel" else "channel")
# the same measures for the other protocols, from the tables their runners write next to these
for prefix in ("embedding", "direct"):
    table = OUT / f"{prefix}_channel_contributions.csv"
    if not table.exists():
        continue
    # their runners label the combined tools channel "Tool calls + outputs"; the figure's row is "Tools"
    other = pd.read_csv(table).rename(columns={"wilcoxon_p": "p"}).replace({"channel": {"Tool calls + outputs": "Tools"}})
    other_fulls = pd.read_csv(OUT / f"{prefix}_channel_contributions_full_transcript.csv").set_index("domain")
    contributions_plots.channel_figure(other, other_fulls, RUNS, OUT, f"{prefix}_channel_contributions", COLOR_BY)

# the share of model-written characters per channel, one pie per domain
frame = pd.DataFrame(shares).T
share = frame.div(frame.sum(axis=1), axis=0)
share.to_csv(OUT / "channel_shares.csv")
print((share * 100).round(1).to_string())
contributions_plots.shares_figure(share, OUT)

# the manuscript figures, as change in squared error from the full transcript: each channel kept alone, then each channel removed. Maths and science
# have no tool calls, and their hidden-reasoning run is not drawn a second time as "only messages".
SPLIT_PARTS = {"Reasoning": ("only-reasoning", "drop-reasoning"), "Messages": ("only-text", "drop-text"),
               "Tools": ("only-tools", "drop-tools")}
PIE_COLORS = contributions_plots.PIE_COLORS
SPLIT_COLOURS = {"Reasoning": PIE_COLORS["CoT"], "Messages": PIE_COLORS["Messages"], "Tools": PIE_COLORS["Tools"]}
contributions_plots.save_figures(
    errors, SPLIT_PARTS, SPLIT_COLOURS, OUT, "channel_contributions", final="transcript_channels",
    caption="Pairwise judge on transcript channels (reasoning, messages, tool calls with their outputs), 40 train "
            "items per semester, 2025-II and 2026-I pooled, each semester's best pairwise judge with reasoning shown. "
            "Maths and Science have no tool calls: the transcript without reasoning is the reasoning-hidden run, and the "
            "transcript without messages is the reasoning alone.")
