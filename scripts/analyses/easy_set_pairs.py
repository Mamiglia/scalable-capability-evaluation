"""EASY set per semester = items the *previous* semester's 3 best models solved on every repeat;
then, per (new model of S) x (old model of S-1) pair -- 8 x 8 = 64 -- the items both got right on repeat 0 (what
`build_dataset` pairs under require_both_correct).

    uv run python scripts/analyses/easy_set_pairs.py science
"""
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/analyses")
from configs.domain import DomainConfig  # noqa: E402
from evals.responses import responses  # noqa: E402
from domain_collection import DOMAINS, ROSTER  # noqa: E402


def main(domain: str):
    out = Path(f"analysis/{domain}_collection")
    _, _, panels = DOMAINS[domain]
    label = {b: lab for lab, bs in panels for b in bs}
    era_of = {m: era for era, ms in ROSTER.items() for m, _ in ms}
    eras = list(ROSTER)
    resp = responses(DomainConfig.from_cli(f"configs/domain/{domain}.yml"))
    resp = resp[resp.model.isin(era_of)].assign(era=lambda d: d.model.map(era_of), bench=lambda d: d.bench.map(label))
    resp = resp.dropna(subset=["bench"])
    benches = [lab for lab, _ in panels if lab in set(resp.bench)]

    per_model = resp.groupby(["era", "bench", "item", "model"]).correct.mean().reset_index()
    best = (per_model.groupby(["era", "model"]).correct.mean().reset_index()
            .sort_values("correct", ascending=False).groupby("era").head(3))
    top3 = per_model.merge(best[["era", "model"]]).groupby(["era", "bench", "item"]).correct.min()
    easy = {era: set(top3[era][top3[era] >= 1].index) for era in eras if era in top3.index.get_level_values(0)}
    r0 = resp[resp.repeat == 0].set_index(["model", "bench", "item"]).correct

    sizes, pairs = [], []
    for prev, era in zip(eras, eras[1:]):
        items = easy[prev]
        sizes.append(dict(semester=era, easy_from=prev, **{b: sum(1 for bb, _ in items if bb == b) for b in benches}, total=len(items)))
        for a, b in product([m for m, _ in ROSTER[era]], [m for m, _ in ROSTER[prev]]):
            row = dict(semester=era, new_model=a, old_model=b)
            for bench in benches:
                its = [i for bb, i in items if bb == bench]
                row[bench] = sum(1 for i in its if r0.get((a, bench, i), False) and r0.get((b, bench, i), False))
            row["total"] = sum(row[b] for b in benches)
            pairs.append(row)
    sizes = pd.DataFrame(sizes)
    pairs = pd.DataFrame(pairs)
    sizes.to_csv(out / "easy_set_sizes.csv", index=False)
    pairs.to_csv(out / "easy_pairs.csv", index=False)
    print("3 best models per semester (define the next semester's EASY set):")
    print(best.groupby("era").model.apply(list).to_string())
    print("\nEASY set size per semester (previous semester's top3 all-correct items):")
    print(sizes.to_string(index=False))
    summ = pairs.groupby("semester")[benches + ["total"]].agg(["mean", "min", "max"]).round(1)
    summ.to_csv(out / "easy_pairs_summary.csv")
    print("\nboth-correct items per pair (repeat 0), mean/min/max over the 64 new x old pairs of each semester:")
    print(summ.to_string())
    print("\nper pair:")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "science")
