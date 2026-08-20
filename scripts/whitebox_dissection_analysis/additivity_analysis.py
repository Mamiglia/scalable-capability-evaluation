"""Summarise an additivity-probe run: magnitude and interaction versus k.

Per k, pooled over prompts and triples:
- median |delta| of the singles (the magnitude-versus-k curve);
- additivity residual e = delta_union - (delta_a + delta_b): median |e|,
  the ratio median|e| / median|delta_a + delta_b|, and R² of delta_union
  predicted by the sum (no fitting).
"""
import sys

import numpy as np
import pandas as pd

df = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
print(f"{df.groupby(['pair_uid', 'order']).ngroups} prompts, "
      f"{df.groupby('k').size().iloc[0]} triples per k per prompt")

rows = []
for k, g in df.groupby("k"):
    singles = np.concatenate([g["delta_a"], g["delta_b"]])
    s = g["delta_a"] + g["delta_b"]
    e = g["delta_union"] - s
    r2 = 1 - np.sum(e**2) / np.sum((g["delta_union"] - g["delta_union"].mean())**2)
    rows.append({"k": k, "median_abs_single": np.median(np.abs(singles)),
                 "q90_abs_single": np.quantile(np.abs(singles), 0.9),
                 "median_abs_resid": np.median(np.abs(e)),
                 "resid_over_sum": np.median(np.abs(e)) / np.median(np.abs(s)),
                 "r2_union_from_sum": r2,
                 "corr": np.corrcoef(s, g["delta_union"])[0, 1]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Model-free kill test for ANY monotone link (wb1): if delta = g(additive B)
# with g monotone, then sign(d1) = sign(d2) implies |d12| >= max(|d1|, |d2|).
# Violations beyond the bf16 floor are genuine interaction — no scalar
# nonlinearity can produce them.
FLOOR = 0.06  # bf16 rounding margin on F; revisit against the repeat check
print("\nmonotone-link necessary condition, same-sign triples only:")
rows = []
for k, g in df.groupby("k"):
    same = g[np.sign(g["delta_a"]) == np.sign(g["delta_b"])]
    hi = np.maximum(same["delta_a"].abs(), same["delta_b"].abs())
    depth = hi - same["delta_union"].abs()          # positive = violated
    viol = depth > FLOOR
    rows.append({"k": k, "same_sign_n": len(same),
                 "violation_rate": viol.mean(),
                 "median_violation_depth": depth[viol].median() if viol.any() else 0.0})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Saturating link: delta = g_c(B(S)) with additive B and g_c(x) = c tanh(x/c).
# Then atanh-scale values u = g_c^{-1}(delta) should be additive over disjoint
# sets. Fit one c per prompt by minimising the link-scale additivity error
# pooled over every k; if that removes the k-trend in the deficit, the deficit
# was saturation, not token-token interaction.


def inv_link(y, c):
    return c * np.arctanh(np.clip(y / c, -0.999, 0.999))


def link_error(g, c):
    u = inv_link(g["delta_a"].to_numpy(), c)
    v = inv_link(g["delta_b"].to_numpy(), c)
    w = inv_link(g["delta_union"].to_numpy(), c)
    return w - (u + v)


print("\nlink-scale additivity (one c per prompt, fitted on all k pooled):")
fits = []
for (uid, order), g in df.groupby(["pair_uid", "order"]):
    grid = np.geomspace(g["delta_union"].abs().max() * 1.01, 50, 40)
    sse = [np.sum(link_error(g, c) ** 2) for c in grid]
    fits.append({"pair_uid": uid, "order": order, "c": grid[int(np.argmin(sse))]})
fits = pd.DataFrame(fits)
print("fitted c:", fits["c"].round(2).describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict())

merged = df.merge(fits, on=["pair_uid", "order"])
rows = []
for k, g in merged.groupby("k"):
    e_id = g["delta_union"] - (g["delta_a"] + g["delta_b"])
    e_ln = np.concatenate([link_error(gp, gp["c"].iloc[0])
                           for _, gp in g.groupby(["pair_uid", "order"])])
    s_ln = np.concatenate([inv_link(gp["delta_a"].to_numpy(), gp["c"].iloc[0])
                           + inv_link(gp["delta_b"].to_numpy(), gp["c"].iloc[0])
                           for _, gp in g.groupby(["pair_uid", "order"])])
    rows.append({"k": k, "identity_resid_over_sum":
                 np.median(np.abs(e_id)) / np.median(np.abs(g["delta_a"] + g["delta_b"])),
                 "link_resid_over_sum": np.median(np.abs(e_ln)) / np.median(np.abs(s_ln))})
print(pd.DataFrame(rows).round(3).to_string(index=False))
