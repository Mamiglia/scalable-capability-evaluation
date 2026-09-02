"""Restricted projection analysis: where does each direction's effect live?

Per direction and restriction (full / not_last / role / not_role), the mean
gap change against baseline, the empirical noise floor from zero-touched
passes, and the coupling of each component to the attribution ledger
(predicted removal effect = minus the role's summed sentence scores).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge")
sys.path.insert(0, str(REPO / "judge-whitebox"))
import token_mediation as tm

S = REPO / "tmp/whitebox"


def main(parts_root):
    t = pd.concat([pd.read_parquet(p)
                   for p in sorted(pathlib.Path(parts_root).rglob("part_*.parquet"))],
                  ignore_index=True)
    base = t[t.name == "baseline"].set_index(["pair_uid", "order"])
    t = t[t.name != "baseline"].copy()
    t["dgap"] = t.gap.values - base.gap.reindex(list(zip(t.pair_uid, t.order))).values
    print(f"{len(base)} prompts")

    zero = t[(t.restriction == "role") & (t.n_touched == 0)]
    print(f"noise floor (zero-touched, n={len(zero)}): "
          f"mean dgap {zero.dgap.mean():+.4f}, mean |dgap| {zero.dgap.abs().mean():.4f}")

    scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
    roles, names = tm.role_ids(scores)
    scores = scores.assign(role=[names[r] for r in roles])
    pred = scores.groupby(["pair_uid", "order", "role"]).score.sum().mul(-1) \
        .rename("pred").reset_index()

    d14 = t[(t.layer == 14) & ~((t.restriction == "role") & (t.n_touched == 0))]
    for name in sorted(d14.name.unique()):
        print(f"\n{name}:")
        for restriction in ("full", "not_last", "role", "not_role"):
            s = d14[(d14.name == name) & (d14.restriction == restriction)]
            if s.empty:
                continue
            j = s.merge(pred[pred.role == name], on=["pair_uid", "order"])
            r = np.corrcoef(j.pred, j.dgap)[0, 1] if len(j) > 2 else float("nan")
            print(f"  {restriction:9s}: n {len(s):4d}  mean dgap {s.dgap.mean():+.4f}  "
                  f"|dgap| {s.dgap.abs().mean():.4f}  kl {s.kl_answer.mean():.5f}  "
                  f"touched median {int(s.n_touched.median())}  ledger r {r:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else S / "causal_v2_masked")
