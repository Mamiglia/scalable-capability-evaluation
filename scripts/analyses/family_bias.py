"""Self and in-family preference of the pairwise judges, from the verdicts on disk (no judging is run).

    uv run python scripts/analyses/family_bias.py [--recompute]

Verdicts: every logs/judge/<domain>/<semester>/<judge>/<hash> run on the frozen train items, judges with > 500
verdicts, conditions 7ea04f2b|9c1d3abb|aff79999 (reasoning hidden), 24d6cc32 (shown), 66c170cd (shown, 10 items per
quartile). A model is the judge's SELF, its FAMILY (same vendor, see FAMILY) or NEUTRAL.

Pair level. Row = one judged (item, pair, shown order) and one side m with relation self/family to judge j, opponent
neutral to j. dev = [j picked m] - the mean pick of m by the other judges of the same run that are neutral to both
sides, on the same (item, pair, order). A decisive judge deviates from the consensus in the consensus' direction,
so the control is j's own dev on pairs neutral to j with the consensus binned (0.1); bias = dev - control, in
probability points. Unit of the CI: the judge run (domain, semester, condition, judge); mean +- t 95% over runs.

theta-hat level. Per run the targets are placed by JudgeLinear (`linear`) and pinned BT (`bt`); excess = theta-hat
of a target by j minus its mean theta-hat by the other judges of the run neutral to it; err = theta-hat - theta*.
Counterfactual `noself`: the judge's own transcripts dropped from the reference set (7 references).

Writes analysis/judge/family_bias/{verdicts.parquet (cache), theta.csv, pairs.csv, theta_hat.csv, noself.csv,
family_bias.png|svg}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent))
from plots import INK2, MUTED, PALETTE, legend, save, style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from capability.task import Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols.pairwise import from_logs  # noqa: E402
from protocols.pairwise.bradley_terry import rank  # noqa: E402
from protocols.pairwise.linear import weighted_theta  # noqa: E402

OUT = Path("analysis/judge/family_bias")
DOMAINS = ["science", "math", "coding", "cyber"]
CONDITIONS = {"7ea04f2b": "hidden", "9c1d3abb": "hidden", "aff79999": "hidden", "24d6cc32": "shown", "66c170cd": "shown-10pq"}
RUN = ["domain", "semester", "condition", "judge_name"]
MIN_VERDICTS = 500  # pilot judges (o4-mini, gpt-5.2, ... ~50 verdicts) are skipped
FAMILY = {"deepseek": "deepseek", "gemma": "google", "glm": "zhipu", "gpt": "openai", "o4": "openai", "hermes": "nous",
          "nous": "nous", "kimi": "moonshot", "llama": "meta", "meta-llama": "meta", "minimax": "minimax",
          "mistral": "mistral", "mixtral": "mistral", "open-mistral": "mistral", "mpt": "mosaic", "nvidia": "nvidia",
          "phi": "microsoft", "wizardlm": "microsoft", "qwen": "alibaba", "ling": "ant"}


def family(m: str) -> str:
    return next(v for k, v in FAMILY.items() if m.startswith(k))


def relation(judge: str, m: str) -> str:
    return "self" if m == judge else "family" if family(m) == family(judge) else "neutral"


def load(recompute: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(verdicts on the train items, theta* of references and targets), cached under OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not recompute and (OUT / "verdicts.parquet").exists():
        return pd.read_parquet(OUT / "verdicts.parquet"), pd.read_csv(OUT / "theta.csv")
    verdicts, theta = [], []
    for name in DOMAINS:
        dom = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        for sem_dir in sorted(Path("logs/judge", name).iterdir()):
            task = Task.from_semester(sem_dir.name, dom)
            theta += [(name, sem_dir.name, m, "reference", th) for m, th in task.references.theta.items()]
            theta += [(name, sem_dir.name, m, "target", th) for m, th in task.truth.items()]
            for run in sorted(sem_dir.glob("*/*/")):
                if run.name not in CONDITIONS:
                    continue
                v = from_logs(run).merge(task.references.items[["bench", "item"]])
                v = v.sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
                verdicts.append(v.assign(domain=name, semester=sem_dir.name, judge_name=run.parts[-2], condition=CONDITIONS[run.name]))
    verdicts = pd.concat(verdicts, ignore_index=True)
    theta = pd.DataFrame(theta, columns=["domain", "semester", "model", "role", "theta"])
    verdicts.to_parquet(OUT / "verdicts.parquet")
    theta.to_csv(OUT / "theta.csv", index=False)
    return verdicts, theta


def pair_level(v: pd.DataFrame, theta: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, judge, side model m with relation self/family, neutral opponent, item, order): y, base, bias."""
    side = pd.concat([v.assign(m=v.model_left, o=v.model_right, y=v.pick_left, first=True),
                      v.assign(m=v.model_right, o=v.model_left, y=1 - v.pick_left, first=False)])
    side = side[[*RUN, "bench", "item", "m", "o", "first", "y"]]
    side["rel_m"] = [relation(j, m) for j, m in zip(side.judge_name, side.m)]
    side["rel_o"] = [relation(j, o) for j, o in zip(side.judge_name, side.o)]
    side["neutral_both"] = (side.rel_m == "neutral") & (side.rel_o == "neutral")
    key = ["domain", "semester", "condition", "bench", "item", "m", "o", "first"]
    consensus = side[side.neutral_both].groupby(key).y.agg(["sum", "count"])
    side = side.merge(consensus, left_on=key, right_index=True, how="left")
    own = side.neutral_both.astype(float)  # leave the judge itself out of its consensus
    side["base"] = (side["sum"] - own * side.y) / (side["count"] - own)
    side = side.dropna(subset=["base"])
    side["dev"] = side.y - side.base
    side["bin"] = np.floor(side.base * 10).clip(0, 9)
    control = side[side.neutral_both].groupby([*RUN, "bin"]).dev.mean().rename("control")
    rows = side[(side.rel_m != "neutral") & (side.rel_o == "neutral")].merge(control, left_on=[*RUN, "bin"], right_index=True, how="left")
    rows["bias"] = rows.dev - rows.control
    rows["role"] = theta.set_index(["domain", "semester", "model"]).role.reindex(pd.MultiIndex.from_frame(rows[["domain", "semester", "m"]])).to_numpy()
    return rows.drop(columns=["sum", "count", "neutral_both"])


def place(g: pd.DataFrame, refs: pd.Series, targets: list[str]) -> pd.DataFrame:
    g = g[g.model_left.isin([*refs.index, *targets]) & g.model_right.isin([*refs.index, *targets])]
    bt = rank(g, anchors=refs).set_index("model").ability.reindex(targets).fillna(refs.mean())
    return pd.DataFrame({"linear": weighted_theta(g, refs, targets), "bt": bt})


def theta_level(v: pd.DataFrame, theta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per run x target x conversion: theta_hat, err, excess vs neutral judges; per run: MSE with/without self)."""
    rows, noself = [], []
    for (dom, sem, cond), g in v.groupby(["domain", "semester", "condition"]):
        t = theta[(theta.domain == dom) & (theta.semester == sem)].set_index("model")
        refs, targets = t.theta[t.role == "reference"], list(t.index[t.role == "target"])
        est = {judge: place(gj, refs, targets) for judge, gj in g.groupby("judge_name")}
        for judge, gj in g.groupby("judge_name"):
            if judge in refs.index:
                alt = place(gj[(gj.model_left != judge) & (gj.model_right != judge)], refs.drop(judge), targets)
                for conv in ("linear", "bt"):
                    noself.append(dict(domain=dom, semester=sem, condition=cond, judge=judge, conversion=conv,
                                       mse=((est[judge][conv] - t.theta[targets]) ** 2).mean(), mse_noself=((alt[conv] - t.theta[targets]) ** 2).mean()))
            for m in targets:
                neutral = [k for k in est if k != judge and relation(k, m) == "neutral"]
                for conv in ("linear", "bt"):
                    rows.append(dict(domain=dom, semester=sem, condition=cond, judge=judge, model=m, relation=relation(judge, m), conversion=conv,
                                     theta=t.theta[m], theta_hat=est[judge][conv][m], err=est[judge][conv][m] - t.theta[m],
                                     excess=est[judge][conv][m] - np.mean([est[k][conv][m] for k in neutral]) if neutral else np.nan, n_neutral=len(neutral)))
    return pd.DataFrame(rows), pd.DataFrame(noself)


def ci(x: pd.Series) -> pd.Series:
    """Mean and t 95% CI over the unit rows of x."""
    n = x.notna().sum()
    h = student_t.ppf(0.975, n - 1) * x.std() / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({"n": n, "mean": x.mean(), "lo": x.mean() - h, "hi": x.mean() + h})


def per_run(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pair-level bias per (run, judge, category), category = self / family (references and targets pooled)."""
    return pairs.rename(columns={"rel_m": "cat"}).groupby([*RUN, "cat"]).agg(n=("y", "size"), pick=("y", "mean"), base=("base", "mean"), bias=("bias", "mean")).reset_index()


def figure(runs: pd.DataFrame, th: pd.DataFrame, out: Path):
    """a: pair-level bias per judge run by category (dots) with mean +- 95% CI over runs, colour = judge family.
    b: theta-hat excess vs neutral judges per (run, target), in-family vs neutral targets, JudgeLinear."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    fams = ["moonshot", "deepseek", "zhipu", "minimax", "alibaba", "openai"]  # judge families with >= 4 runs; the rest grey
    colour = {**dict(zip(fams, PALETTE)), "other": MUTED}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.0, 1.9))
    rng = np.random.default_rng(0)
    for i, cat in enumerate(["self", "family"]):
        d = runs[runs.cat == cat]
        fam = d.judge_name.map(family).where(lambda f: f.isin(fams), "other")
        a.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d.bias, s=7, c=fam.map(colour), alpha=0.7, lw=0, zorder=3)
        s = ci(d.bias)
        a.plot([i + 0.3, i + 0.3], [s.lo, s.hi], color=INK2, lw=1, zorder=4)
        a.plot(i + 0.3, s["mean"], "o", ms=4, color=INK2, zorder=5)
    a.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    a.set_xticks(range(2), ["Self", "Family"])
    a.set_ylabel("Pick bias vs neutral judges")
    a.set_ylim(-0.3, 0.3), a.set_yticks([-0.3, 0, 0.3])
    d = th[(th.conversion == "linear") & th.excess.notna()]
    for i, rel in enumerate(["neutral", "family"]):
        e = d[d.relation == rel]
        fam = e.judge.map(family).where(lambda f: f.isin(fams), "other")
        b.scatter(i + rng.uniform(-0.18, 0.18, len(e)), e.excess, s=7, c=fam.map(colour), alpha=0.7, lw=0, zorder=3)
        s = ci(e.excess)
        b.plot([i + 0.3, i + 0.3], [s.lo, s.hi], color=INK2, lw=1, zorder=4)
        b.plot(i + 0.3, s["mean"], "o", ms=4, color=INK2, zorder=5)
    b.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    b.set_xticks(range(2), ["Neutral\ntarget", "Family\ntarget"])
    b.set_ylabel("θ̂ excess vs neutral judges")
    b.set_ylim(-2, 2), b.set_yticks([-2, 0, 2])  # one point (gpt-4-0613, math 2024-I, gpt-4o: +3.1) sits above the range
    for ax in (a, b):
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 5))
        ax.set_xlim(-0.5, len(ax.get_xticks()) - 0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=colour[f], label=f) for f in [*fams, "other"]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, out)


def main(recompute: bool = False):
    v, theta = load(recompute)
    counts = v.groupby(RUN).size()
    v = v[counts.reindex(pd.MultiIndex.from_frame(v[RUN])).to_numpy() > MIN_VERDICTS]
    pairs = pair_level(v, theta)
    runs = per_run(pairs)
    runs.to_csv(OUT / "pairs.csv", index=False, float_format="%.3f")
    th, noself = theta_level(v, theta)
    th.to_csv(OUT / "theta_hat.csv", index=False, float_format="%.3f")
    noself.to_csv(OUT / "noself.csv", index=False, float_format="%.3f")
    figure(runs, th, OUT / "family_bias")

    print("Pair-level bias (probability points), unit = judge run")
    print(runs.groupby("cat").bias.apply(ci).unstack().round(3).to_string())
    print(runs.groupby(["condition", "cat"]).bias.apply(ci).unstack().round(3).to_string())
    print(runs.assign(fam=runs.judge_name.map(family)).groupby(["fam", "cat"]).bias.apply(ci).unstack().round(3).to_string())
    print("\ntheta-hat excess vs neutral judges (theta units), unit = run x target")
    print(th.dropna(subset=["excess"]).groupby(["conversion", "relation"]).excess.apply(ci).unstack().round(3).to_string())
    print(th.dropna(subset=["excess"]).groupby(["conversion", "relation"]).err.apply(ci).unstack().round(3).rename(columns={"mean": "err"}).to_string())
    print("\nMSE with the judge's own transcripts as a reference vs dropped, unit = run")
    print(noself.groupby("conversion")[["mse", "mse_noself"]].mean().round(3).to_string())


if __name__ == "__main__":
    main("--recompute" in sys.argv)
