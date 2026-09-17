"""Effect of the number of reference models on the pairwise judge's prediction score, per domain and semester (no judging is run).

    uv run python scripts/analyses/reference_count_effect.py [--recompute] [--last K]

Same verdicts and prediction score as item_count_effect.py (the top-1 reference as judge, reasoning shown, all judged
train items). For k in 2, 4, 6 and all references, EVERY k-subset of the references is taken as the whole reference
set: only its theta* and its verdicts (subset-vs-subset for the calibration, target-vs-subset for the placement) are
kept, and the targets are placed by JudgeLinear (`linear`, the figure) and pinned BT (`bt`). `mean_sub` is the
constant subset mean, which JudgeLinear falls back to when a bench has fewer than 3 reference pairs (always at k=2).
MSE as % of the ALL-reference mean_ref baseline, so scores are comparable across k; prediction score = 100 - that.
Band = 2.5-97.5 percentile over subsets (which references were kept). What was removed is read off the subsets: the
number of frontier references (fit.yml `top`) removed at k = 4 and 6 (`_frontier` figure), and the subsets of each k split by the
mean theta* of the references they keep: top third = keep stronger, bottom third = keep weaker (`_ends` figure, median
and 2.5-97.5 percentile band over the subsets of that third, i.e. over which models were removed), plus the Spearman rho between the removed references' mean
theta* and the score. Writes analysis/judge/reference_count_effect{,_frontier,_ends}.{png,svg} and the .csv.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from item_count_effect import CONDITIONS, DOMAINS, plot  # noqa: E402

from capability.evaluate import score  # noqa: E402
from capability.task import Task  # noqa: E402
from configs.domain import DomainConfig  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402
from protocols.pairwise import from_logs  # noqa: E402
from protocols.pairwise.bradley_terry import rank  # noqa: E402
from protocols.pairwise.linear import weighted_theta  # noqa: E402

OUT = Path("analysis/judge/reference_count_effect")
SIZES = [2, 4, 6]
KEYS = ["linear", "bt", "mean_sub"]


def forecasts(verdicts: pd.DataFrame, refs: pd.Series, targets: list[str]) -> dict[str, pd.Series]:
    """Place the targets with `refs` as the whole reference set."""
    v = verdicts[verdicts.model_left.isin([*refs.index, *targets]) & verdicts.model_right.isin([*refs.index, *targets])]
    bt = rank(v, anchors=refs).set_index("model").ability.reindex(targets).fillna(refs.mean())
    return {"linear": weighted_theta(v, refs, targets), "bt": bt, "mean_sub": pd.Series(refs.mean(), index=targets)}


def judged():
    """Per domain and semester: the task, its mean_ref MSE, the top-1 judge and its verdicts on the train items."""
    for name in DOMAINS:
        domain = DomainConfig.from_cli(Path(f"configs/domain/{name}.yml"))
        top = yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"]
        for sem, judges in top.items():
            judge, task = judges[0].lower(), Task.from_semester(sem, domain)
            run = next((p for c in CONDITIONS if (p := Path("logs/judge", name, sem, judge, c)).exists()), None)
            if run is None:
                print("missing", name, sem, judge, file=sys.stderr)
                continue
            v = from_logs(run).sort_values("imported", kind="stable").drop_duplicates(["bench", "item", "model_left", "model_right"])
            print(name, sem, judge, file=sys.stderr)
            yield dict(domain=name, semester=sem, judge=judge, judge_rank=1, baseline=score(PROTOCOLS["mean_ref"](task), task.truth)["mse"]), \
                task, v.merge(task.references.items[["bench", "item"]])


def compute() -> pd.DataFrame:
    rows = []
    for cell, task, v in judged():
        refs, targets = task.references.theta, list(task.test.models.index)
        for k in [s for s in SIZES if s < len(refs)] + [len(refs)]:
            for subset in combinations(refs.index, k):
                mse = {key: score(f, task.truth)["mse"] for key, f in forecasts(v, refs[list(subset)], targets).items()}
                rows.append(dict(**cell, n_refs=len(refs), k=k, subset="|".join(subset), **mse))
    return pd.DataFrame(rows)


def removed(df: pd.DataFrame) -> pd.DataFrame:
    """Per subset, what was removed: how many frontier references (fit.yml `top`), and the capability of the removed ones."""
    out = []
    for (name, sem), cell in df.groupby(["domain", "semester"]):
        models = pd.read_csv(f"data/meta/{name}/models.csv", index_col="model")
        theta = models.loc[models[f"{sem}_role"] == "reference", f"{sem}_theta"].sort_values()
        frontier = {m.lower() for m in yaml.safe_load(Path(f"data/meta/{name}/fit.yml").read_text())["top"][sem]}
        gone = cell.subset.map(lambda kept: theta.drop(kept.split("|")))
        out.append(cell.assign(frontier_removed=gone.map(lambda g: len(frontier & set(g.index))),
                               theta_removed=gone.map(lambda g: g.mean()),  # NaN when nothing is removed
                               theta_kept=cell.subset.map(lambda kept: theta[kept.split("|")].mean())))
    out = pd.concat(out)
    # the subsets of a task and k, by the capability of what they keep: the top third keeps the stronger references
    third = out.groupby(["domain", "semester", "k"]).theta_kept.rank(pct=True)
    return out.assign(keep=np.where(third > 2 / 3, "stronger", np.where(third <= 1 / 3, "weaker", "")))


def main() -> int:
    csv = OUT.with_suffix(".csv")
    if "--recompute" in sys.argv or not csv.exists():
        compute().to_csv(csv, index=False, float_format="%.4f")
    df = pd.read_csv(csv)
    df = removed(df[df.judge_rank == 1])  # the strongest judge only
    df["score"] = 100 - 100 * df.linear / df.baseline
    cells = ["domain", "semester"]
    for key in KEYS:
        draws = (100 - 100 * df[key] / df.baseline).groupby([df.domain, df.semester, df.k])
        print(f"\n{key}: median prediction score over subsets [2.5, 97.5 percentile]")
        table = draws.median().round(0).astype(int).astype(str) + " [" + draws.quantile(0.025).round(0).astype(int).astype(str) \
            + ", " + draws.quantile(0.975).round(0).astype(int).astype(str) + "]"
        print(table.unstack("k").to_string())
    some = df[df.k.isin([4, 6])]
    print("\nlinear: median prediction score by number of frontier references removed (columns: k kept, frontier removed)")
    print(some.groupby([*cells, "k", "frontier_removed"]).score.median().round(0).unstack(["k", "frontier_removed"]).sort_index(axis=1).to_string())
    print("\nlinear: prediction score of the third of the subsets keeping the stronger / weaker references, median [2.5, 97.5 percentile]")
    full = df[df.k == df.n_refs]  # at all references the two meet
    ends = pd.concat([full.assign(keep="stronger"), df[df.keep == "stronger"], full.assign(keep="weaker"), df[df.keep == "weaker"]])
    draws = ends.groupby([*cells, "keep", "k"]).score
    table = draws.median().round(0).astype(int).astype(str) + " [" + draws.quantile(0.025).round(0).astype(int).astype(str) \
        + ", " + draws.quantile(0.975).round(0).astype(int).astype(str) + "]"
    print(table.unstack(["keep", "k"]).sort_index(axis=1).to_string())
    print("\nlinear: Spearman rho over subsets between the removed references' mean theta* and the prediction score")
    rho = some.groupby([*cells, "k"]).apply(lambda g: g.theta_removed.corr(g.score, method="spearman"))
    print(rho.round(2).unstack("k").to_string())
    tag = ""
    if "--last" in sys.argv:  # only the last K semesters of every domain
        n = int(sys.argv[sys.argv.index("--last") + 1])
        last, tag = sorted(df.semester.unique())[-n:], f"_last{n}"
        df, some, ends = (d[d.semester.isin(last)] for d in (df, some, ends))
    shared = dict(tag=tag, xlabel="Reference models", xticks=(2, 4, 6, 8))
    plot(df, x="k", out=OUT, **shared)
    plot(ends.assign(row="Keep " + ends.keep), x="k", out=OUT.with_name(OUT.name + "_ends"), **shared)
    plot(some.sort_values("k", ascending=False, kind="stable").assign(row=lambda d: d.k.astype(str) + " kept"), tag=tag, x="frontier_removed",
         xlabel="Frontier references removed", xticks=(0, 1, 2, 3), out=OUT.with_name(OUT.name + "_frontier"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
