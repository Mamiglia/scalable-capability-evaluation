"""Causal projection analysis: effect vs random null, dose curve, layer
specificity, selectivity, predicted-vs-observed.

Observed effect of a condition is its gap minus the same prompt's baseline
gap. The null for each fitted direction is the 20 random unit directions at
the same layer and dose. Predicted effect of removing a role direction is
minus the summed sentence scores of that role's sentences.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge")
sys.path.insert(0, str(REPO / "judge-whitebox"))
import token_mediation as tm

S = REPO / "tmp/whitebox"
FIT_LAYER, CONTROL_LAYER = 14, 21
SEED = 0
BOOT = 2000


def item_boot_mean(values, items, seed=SEED, n=BOOT):
    """Mean with a 95% interval and a one-sided p for mean > 0, items resampled."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(items)
    groups = {it: values[items == it] for it in uniq}
    means = np.array([np.concatenate([groups[it] for it in rng.choice(uniq, len(uniq),
                                                                      replace=True)]).mean()
                      for _ in range(n)])
    p_pos = (np.sum(means <= 0) + 1) / (n + 1)
    return float(np.mean(values)), tuple(np.percentile(means, [2.5, 97.5])), float(p_pos)


def item_boot_corr(a, b, items, seed=SEED, n=BOOT):
    rng = np.random.default_rng(seed)
    uniq = np.unique(items)
    idx = {it: np.flatnonzero(items == it) for it in uniq}
    rs = []
    for _ in range(n):
        draw = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx[it] for it in draw])
        rs.append(np.corrcoef(a[sel], b[sel])[0, 1])
    return float(np.corrcoef(a, b)[0, 1]), tuple(np.percentile(rs, [2.5, 97.5]))


def predicted_effects(scores):
    """Minus the summed sentence scores per (prompt, role), roles as in the fit."""
    roles, names = tm.role_ids(scores)
    scores = scores.assign(role=[names[r] for r in roles])
    pred = scores.groupby(["pair_uid", "order", "item", "role"]).score.sum().mul(-1)
    return pred.reset_index()


def main(parts_root):
    t = tm.pd.concat([pd.read_parquet(p)
                      for p in sorted(pathlib.Path(parts_root).rglob("part_*.parquet"))],
                     ignore_index=True)
    base = t[t.name == "baseline"].set_index(["pair_uid", "order"])
    t = t[t.name != "baseline"].copy()
    key = list(zip(t.pair_uid, t.order))
    t["dgap"] = t.gap.values - base.gap.reindex(key).values
    t["dnll"] = t.nll.values - base.nll.reindex(key).values
    n_prompts = base.shape[0]
    print(f"{n_prompts} prompts")

    directions = sorted(n for n in t.name.unique() if not n.startswith("random"))
    rand = t[t.name.str.startswith("random") & (t.layer == FIT_LAYER)]
    rand_means = rand.groupby("name").dgap.apply(lambda g: g.abs().mean())
    print(f"\nrandom null (n=20): mean |dgap| {rand_means.mean():.4f} "
          f"[{rand_means.min():.4f}, {rand_means.max():.4f}], "
          f"kl {rand.kl_answer.mean():.5f}, dnll {rand.dnll.mean():.5f}")

    scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
    pred = predicted_effects(scores)

    rand_prompt = rand.groupby(["pair_uid", "order"]).dgap.apply(lambda g: g.abs().mean())

    print("\nper direction, layer 14 dose 1 (paired against the 20-random null):")
    results = []
    for name in directions:
        d = t[(t.name == name) & (t.layer == FIT_LAYER) & (t.dose == 1.0)]
        eff = float(d.dgap.abs().mean())
        rank_p = float((rand_means >= eff).sum() + 1) / 21
        paired = d.dgap.abs().to_numpy() \
            - rand_prompt.reindex(list(zip(d.pair_uid, d.order))).to_numpy()
        excess, exc_ci, p = item_boot_mean(paired, d.item.to_numpy())
        mean, ci, _ = item_boot_mean(d.dgap.to_numpy(), d.item.to_numpy())
        doses = t[(t.name == name) & (t.layer == FIT_LAYER)].groupby("dose").dgap.mean()
        ctrl = t[(t.name == name) & (t.layer == CONTROL_LAYER)]
        pj = d.merge(pred[pred.role == name], on=["pair_uid", "order", "item"])
        if len(pj) >= 10:
            r, rci = item_boot_corr(pj.score.to_numpy(), pj.dgap.to_numpy(),
                                    pj.item.to_numpy())
            corr = f"pred-vs-obs r {r:.3f} [{rci[0]:.3f}, {rci[1]:.3f}]"
        else:
            corr = "pred-vs-obs: too few rows"
        mono = doses.sort_index()
        monotone = bool(np.all(np.diff(np.abs(mono.to_numpy())) >= -1e-9))
        print(f"  {name}: mean dgap {mean:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], "
              f"|dgap| {eff:.4f}, excess over random {excess:+.4f} "
              f"[{exc_ci[0]:+.4f}, {exc_ci[1]:+.4f}] (boot p {p:.4f}; rank p {rank_p:.3f}), "
              f"dose curve {[round(v, 4) for v in mono.tolist()]} "
              f"{'monotone' if monotone else 'NOT monotone'}, "
              f"L{CONTROL_LAYER} |dgap| {ctrl.dgap.abs().mean():.4f}, "
              f"kl {d.kl_answer.mean():.5f}, dnll {d.dnll.mean():.5f}, {corr}")
        results.append({"name": name, "p_vs_null": p})

    reject = benjamini_hochberg([r["p_vs_null"] for r in results], 0.05)
    print("\nBH 5%:", {r["name"]: bool(k) for r, k in zip(results, reject)})


def benjamini_hochberg(pvals, q):
    p = np.asarray(pvals)
    order = np.argsort(p)
    thresh = q * (np.arange(len(p)) + 1) / len(p)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(len(p), bool)
    out[order[:k]] = True
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else S / "causal_v1")
