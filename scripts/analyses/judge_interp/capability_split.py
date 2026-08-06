"""Split the preference signal into capability and the rest.

Regresses elo on the author model's benchmark logit (item-centered) to
split it into a capability part and a residual, then fits one probe
per part at the best (layer, pooling): v_cap ("author seems capable")
and v_pref ("preferred beyond capability"). Reports held-out r for
both, the angles between v_cap, v_pref and the verdict-trained
approval direction, and how much of the approval score is capability.
The v_cap/approval angle doubles as an internal judge-quality measure:
narrow means the judge rewards ability, wide means it rewards
something else. Control: label-shuffle nulls.

Writes out_dir/capability_directions.npz (v_cap, v_pref, and v_pref
with its v_cap component projected out, plus feature stats).

Usage:
  uv run python scripts/analyses/judge_interp/capability_split.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from probes import fit_one

ROOT = Path(__file__).resolve().parents[3]


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]

    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    best = pd.read_csv(out / "probe_metrics.csv").sort_values("r").iloc[-1]
    li = meta["layers"].index(int(best.layer))
    pi = meta["poolings"].index(best.pooling)
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    features = np.asarray(acts[:, li, pi, :], dtype=np.float32)

    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    y = elo["elo"].to_numpy()
    groups = elo["item"].to_numpy()

    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    cap = elo["model"].map(capability).to_numpy()
    cap = cap - pd.Series(cap).groupby(elo["item"]).transform("mean").to_numpy()
    slope = float(np.polyfit(cap, y, 1)[0])
    resid = y - slope * cap
    print(f"layer {int(best.layer)} {best.pooling}; "
          f"elo ~ capability slope {slope:.3f}, corr {np.corrcoef(cap, y)[0,1]:.3f}")

    rng = np.random.default_rng(pc["seed"])
    results = {}
    for name, target in (("v_cap", cap), ("v_pref", resid)):
        preds, cos, _ = fit_one(features, target, groups, pc["alpha_grid"],
                                pc["cv_folds"], pc["seed"])
        r = float(np.corrcoef(preds, target)[0, 1])
        shuffled = target.copy()
        for item in np.unique(groups):
            m = groups == item
            shuffled[m] = rng.permutation(shuffled[m])
        preds0, _, _ = fit_one(features, shuffled, groups, pc["alpha_grid"],
                               pc["cv_folds"], pc["seed"])
        r0 = float(np.corrcoef(preds0, shuffled)[0, 1])
        results[name] = r
        print(f"{name}: held-out r {r:.3f}  fold-cosine {cos:.3f}  "
              f"shuffle null r {r0:.3f}")

    # Full-data directions in standardized space.
    from sklearn.linear_model import Ridge
    mean, std = features.mean(0), features.std(0) + 1e-8
    z = (features - mean) / std
    alpha = pc["alpha_grid"][-1]
    v_cap = Ridge(alpha).fit(z, cap).coef_
    v_pref = Ridge(alpha).fit(z, resid).coef_
    v_pref_orth = v_pref - (unit(v_cap) @ v_pref) * unit(v_cap)

    bt = np.load(out / "bt_direction.npz")
    w_bt, w_qwen = unit(bt["w"]), unit(bt["w_qwen"])
    print(f"\nangles (cosine): cap-pref {float(unit(v_cap) @ unit(v_pref)):.3f}  "
          f"cap-approval {float(unit(v_cap) @ w_bt):.3f}  "
          f"pref-approval {float(unit(v_pref) @ w_bt):.3f}")
    print(f"judge-quality reading: approval-capability alignment "
          f"{float(unit(v_cap) @ w_bt):.3f} (shared), "
          f"{float(unit(v_cap) @ w_qwen):.3f} (qwen-only)")
    score_bt = z @ bt["w"]
    print(f"approval score: corr with capability {np.corrcoef(score_bt, cap)[0,1]:.3f}, "
          f"with residual preference {np.corrcoef(score_bt, resid)[0,1]:.3f}")

    np.savez(out / "capability_directions.npz", v_cap=v_cap, v_pref=v_pref,
             v_pref_orth=v_pref_orth, mean=mean, std=std,
             layer=int(best.layer), pooling=best.pooling)
    print(f"-> {out / 'capability_directions.npz'}")


if __name__ == "__main__":
    main()
