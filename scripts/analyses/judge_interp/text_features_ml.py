"""Named surface features vs the 5 corrected judge directions.

For every recorded entity (row-aligned to acts), project its activation onto
each of the 5 directions in directions_multilayer.npz (score = z.v_hat, z the
per-direction standardized last-token activation at that direction's layer),
then regress each projection on standardized, item-centred surface features
(text_features.features + boxed-answer presence). The features that raise
v_pref_gen_orth (taste beyond capability) but not v_cap are what judges reward
past what author ability warrants.

Item-centres features and target within each direction's row set (the judge's
comparison is within an item); v_pref_spec* restricted to rows with qwen elo.
Coefficients from RidgeCV (standardized, joint fit); significance from an
item-block bootstrap; R2 is item-grouped held-out (probes.fit_one).

Writes out_dir/text_features_ml.csv.

Usage:
  uv run python scripts/analyses/judge_interp/text_features_ml.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge, RidgeCV

from probes import load_aligned, fit_one
from text_features import features

ROOT = Path(__file__).resolve().parents[3]
NAMES5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]


def unit(v):
    return v / np.linalg.norm(v)


def direction_score(acts, pack, name, layers, poolings):
    """score = ((act@layer,pool) - mean)/std . unit(dir), per entity."""
    li = layers.index(int(pack[f"sl_{name}_layer"]))
    pi = poolings.index(str(pack[f"sl_{name}_pooling"]))
    z = (np.asarray(acts[:, li, pi, :], np.float32)
         - pack[f"sl_{name}_mean"]) / pack[f"sl_{name}_std"]
    return z @ unit(pack[f"sl_{name}_dir"])


def centre(v, groups):
    return v - pd.Series(v).groupby(groups).transform("mean").to_numpy()


def boot_se(x, y, alpha, groups, n_boot, seed):
    """Item-block bootstrap SE of the standardized ridge coefficients."""
    rng = np.random.default_rng(seed)
    items = np.unique(groups)
    idx_by_item = {it: np.where(groups == it)[0] for it in items}
    coefs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_item[it]
                              for it in rng.choice(items, len(items), replace=True)])
        xb = (x[idx] - x[idx].mean(0)) / (x[idx].std(0) + 1e-9)
        yb = (y[idx] - y[idx].mean()) / (y[idx].std() + 1e-9)
        coefs.append(Ridge(alpha).fit(xb, yb).coef_)
    return np.std(coefs, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]

    acts, elo, meta = load_aligned(cfg, out)
    layers, poolings = meta["layers"], meta["poolings"]
    groups = elo["item"].to_numpy()
    pack = np.load(out / "directions_multilayer.npz", allow_pickle=True)

    # ---- direction scores (n, 5) -------------------------------------------
    scores = {name: direction_score(acts, pack, name, layers, poolings)
              for name in NAMES5}

    # anchor: v_cap projection must track author capability (else layer/sign bug)
    logit = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    cap_corr = float(np.corrcoef(scores["v_cap"],
                                 elo["model"].map(logit).to_numpy())[0, 1])
    assert cap_corr > 0.5, f"v_cap score vs author capability corr {cap_corr:.3f}"
    print(f"anchor: corr(v_cap score, author capability) = {cap_corr:+.3f}")

    # row mask per direction: spec ones only where qwen elo exists
    spec = pd.read_csv(out / "elo_qwen3-32b.csv").set_index(["text_id", "item"])["elo"]
    have = np.array([(t, i) in spec.index
                     for t, i in zip(elo["text_id"], elo["item"])])
    mask_of = {n: (have if "spec" in n else np.ones(len(elo), bool)) for n in NAMES5}

    # ---- features ----------------------------------------------------------
    texts = pd.read_parquet(ROOT / cfg["texts_parquet"]).set_index("text_id")["text"]
    def feats(t):
        d = features(texts[t])
        d["boxed answer"] = float(bool(re.search(r"\\boxed", texts[t])))
        return d
    df = pd.DataFrame([feats(t) for t in elo["text_id"]], index=elo.index)
    feat_names = list(df.columns)
    F = df.to_numpy(float)
    print(f"{len(elo)} entities, {len(feat_names)} features; "
          f"{have.sum()} with qwen elo")

    # ---- per-direction regression ------------------------------------------
    coef_tbl, sig_tbl, r2 = {}, {}, {}
    for name in NAMES5:
        m = mask_of[name]
        g = groups[m]
        x = np.column_stack([centre(F[m, j], g) for j in range(F.shape[1])])
        x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        y = centre(scores[name][m], g)
        yz = (y - y.mean()) / (y.std() + 1e-9)

        model = RidgeCV(alphas=pc["alpha_grid"]).fit(x, yz)
        alpha = float(model.alpha_)
        coef_tbl[name] = model.coef_
        se = boot_se(x, yz, alpha, g, args.n_boot, pc["seed"])
        sig_tbl[name] = np.abs(model.coef_) > 1.96 * se

        preds, _, _ = fit_one(x, yz, g, pc["alpha_grid"], pc["cv_folds"], pc["seed"])
        r2[name] = 1 - float(np.mean((preds - yz) ** 2)) / float(np.var(yz))

    coef = pd.DataFrame(coef_tbl, index=feat_names)
    sig = pd.DataFrame(sig_tbl, index=feat_names)
    coef = coef.reindex(coef["v_pref_gen_orth"].abs().sort_values(ascending=False).index)
    sig = sig.reindex(coef.index)

    # ---- report ------------------------------------------------------------
    def cell(v, s):
        return f"{v:+.3f}{'*' if s else ' '}"
    print("\nstandardized ridge coefficients (* = |coef| > 1.96 boot-SE)")
    print(f"{'feature':22s}" + "".join(f"{n:>18}" for n in NAMES5))
    for f in coef.index:
        print(f"{f:22s}" + "".join(f"{cell(coef.loc[f, n], sig.loc[f, n]):>18}"
                                   for n in NAMES5))

    print("\nheld-out R2 (item-grouped CV):")
    for n in NAMES5:
        print(f"  {n:18s} {r2[n]:+.3f}")

    # key divergences: taste beyond capability vs calibrated-on-ability
    cap, taste = "v_cap", "v_pref_gen_orth"
    print("\nKEY DIVERGENCE  (raise v_pref_gen_orth = rewarded beyond ability):")
    print("  taste-positive but NOT capability-positive:")
    for f in coef.index:
        ct, cc = coef.loc[f, taste], coef.loc[f, cap]
        st, sc = sig.loc[f, taste], sig.loc[f, cap]
        if st and ct > 0 and not (sc and cc > 0):
            print(f"    {f:22s} taste {ct:+.3f}*  cap {cc:+.3f}{'*' if sc else ''}")
    print("  taste-negative but NOT capability-negative:")
    for f in coef.index:
        ct, cc = coef.loc[f, taste], coef.loc[f, cap]
        st, sc = sig.loc[f, taste], sig.loc[f, cap]
        if st and ct < 0 and not (sc and cc < 0):
            print(f"    {f:22s} taste {ct:+.3f}*  cap {cc:+.3f}{'*' if sc else ''}")
    print("  capability-positive but NOT taste (judges calibrated on ability):")
    for f in coef.index:
        ct, cc = coef.loc[f, taste], coef.loc[f, cap]
        st, sc = sig.loc[f, taste], sig.loc[f, cap]
        if sc and cc > 0 and not (st and ct > 0):
            print(f"    {f:22s} cap {cc:+.3f}*  taste {ct:+.3f}{'*' if st else ''}")

    # combined table: coef with sig flag, plus R2 row
    tbl = coef.copy()
    for n in NAMES5:
        tbl[f"{n}_sig"] = sig[n].astype(int)
    tbl.loc["_heldout_r2"] = {**{n: r2[n] for n in NAMES5},
                              **{f"{n}_sig": np.nan for n in NAMES5}}
    tbl.to_csv(out / "text_features_ml.csv")
    print(f"\n-> {out / 'text_features_ml.csv'}")


if __name__ == "__main__":
    main()
