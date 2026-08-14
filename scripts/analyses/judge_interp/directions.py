"""The five directions the analysis reads and steers along.

Each is a ridge probe from the judge's layer-26 activations to one
target. Nothing is subtracted from a target: the two orthogonalised
variants remove the capability direction from the fitted direction
afterwards, so `v_pref_gen` and `v_pref_gen_orth` differ only by that
projection and can be compared.

  v_cap             author model's GPQA accuracy, item-centred
  v_pref_gen        pooled elo (all judges)
  v_pref_gen_orth   pooled elo, orthogonalised to v_cap
  v_pref_spec       judge-specific elo (one judge's verdicts)
  v_pref_spec_orth  judge-specific elo, orthogonalised to v_cap

Held-out r is reported per target with a label-shuffle null, and the
angles between the directions are printed, since a small angle between
v_cap and a preference direction is the whole reason the orthogonalised
variants exist.

Writes out_dir/directions.npz (directions + feature stats + layer).

Usage:
  uv run python scripts/analyses/judge_interp/directions.py [--judge qwen3-32b]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from probes import fit_one

ROOT = Path(__file__).resolve().parents[3]


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--judge", default="qwen3-32b",
                    help="the recorded judge; its elo file must already exist")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]

    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    best = pd.read_csv(out / "probe_metrics.csv").sort_values("r").iloc[-1]
    layer, pooling = int(best.layer), str(best.pooling)
    li, pi = meta["layers"].index(layer), meta["poolings"].index(pooling)
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    features = np.asarray(acts[:, li, pi, :], dtype=np.float32)

    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    groups = elo["item"].to_numpy()

    # judge-specific elo covers fewer entities; align by (text_id, item)
    spec = pd.read_csv(out / f"elo_{args.judge}.csv").set_index(["text_id", "item"])["elo"]
    spec_vals = np.array([spec.get((t, i), np.nan)
                          for t, i in zip(elo["text_id"], elo["item"])])
    have = ~np.isnan(spec_vals)

    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    cap = elo["model"].map(capability).to_numpy(dtype=float)
    cap = cap - pd.Series(cap).groupby(elo["item"]).transform("mean").to_numpy()

    targets = {
        "v_cap": (cap, np.ones(len(cap), bool)),
        "v_pref_gen": (elo["elo"].to_numpy(), np.ones(len(cap), bool)),
        "v_pref_spec": (np.nan_to_num(spec_vals), have),
    }
    print(f"layer {layer} {pooling}; {len(elo)} entities, "
          f"{have.sum()} with {args.judge} verdicts")

    rng = np.random.default_rng(pc["seed"])
    fitted, scores = {}, {}
    for name, (y, mask) in targets.items():
        r, _, _ = fit_one(features[mask], y[mask], groups[mask],
                          pc["alpha_grid"], pc["cv_folds"], pc["seed"])
        held = float(np.corrcoef(r, y[mask])[0, 1])
        shuffled = y[mask].copy()
        for item in np.unique(groups[mask]):
            m = groups[mask] == item
            shuffled[m] = rng.permutation(shuffled[m])
        r0, _, _ = fit_one(features[mask], shuffled, groups[mask],
                           pc["alpha_grid"], pc["cv_folds"], pc["seed"])
        null = float(np.corrcoef(r0, shuffled)[0, 1])
        scores[name] = (held, null)
        print(f"{name:16s} held-out r {held:+.3f}   shuffle null {null:+.3f}   "
              f"(n={mask.sum()})")

        mean, std = features[mask].mean(0), features[mask].std(0) + 1e-8
        z = (features[mask] - mean) / std
        fitted[name] = Ridge(pc["alpha_grid"][-1]).fit(z, y[mask]).coef_

    v_cap = fitted["v_cap"]
    fitted["v_pref_gen_orth"] = (fitted["v_pref_gen"]
                                 - (unit(v_cap) @ fitted["v_pref_gen"]) * unit(v_cap))
    fitted["v_pref_spec_orth"] = (fitted["v_pref_spec"]
                                  - (unit(v_cap) @ fitted["v_pref_spec"]) * unit(v_cap))

    names = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]
    print("\nangles (cosine):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a:16s} {b:16s} {float(unit(fitted[a]) @ unit(fitted[b])):+.3f}")

    # one standardisation for everything downstream: the full entity set
    mean, std = features.mean(0), features.std(0) + 1e-8
    rng2 = np.random.default_rng(cfg["interp_pod"]["seed"])
    extras = dict(random_a=rng2.standard_normal(features.shape[1]),
                  random_b=rng2.standard_normal(features.shape[1]))
    np.savez(out / "directions.npz", layer=layer, pooling=pooling,
             mean=mean, std=std, judge=args.judge,
             prompt=cfg["record"]["prompt"],   # pod scripts read the pack directly
             median_norm=float(np.median(np.linalg.norm(features, axis=1))),
             names=names + list(extras),
             **{f"dir_{k}": v for k, v in {**{n: fitted[n] for n in names}, **extras}.items()},
             **{f"r_{n}": scores.get(n, (np.nan, np.nan))[0] for n in names})
    print(f"\n-> {out / 'directions.npz'}")


if __name__ == "__main__":
    main()
