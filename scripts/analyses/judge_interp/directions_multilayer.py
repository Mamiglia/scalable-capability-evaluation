"""Multi-layer judge directions: per-layer and 5-layer-block ridge fits.

Extends directions.py from one layer to every recorded layer plus jointly-fit
5-layer blocks. Three RAW base targets (capability, pooled elo, qwen elo);
nothing is ever removed from a target. Capability is projected out of the
fitted preference DIRECTIONS afterwards (the *_orth variants).

Per base target, both poolings:
  - single-layer sweep: held-out r at every recorded layer; best (layer,
    pooling) is that target's single-layer direction.
  - block sweep: held-out r over sliding windows of 5 consecutive recorded
    layers (concatenated features); best window is the block direction, a
    (5, 5120) object with one joint gap g_block.

For each final direction: g (weak->strong author-capability gap along the
unit direction, the alpha unit), loading validation (projection-vs-target
correlation + a decile-monotone check), matched-norm random controls.

Writes out_dir/directions_multilayer.npz: rich per-(dir, mode) content plus a
legacy flat pod pack (all five single-layer directions refit at one shared
layer so pod_*.py can read layer/mean/std/dir_<name> coherently).

Usage:
  uv run python scripts/analyses/judge_interp/directions_multilayer.py [--judge qwen3-32b] [--quick]
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
BLOCK = 5  # recorded layers per joint block


def unit(v):
    return v / np.linalg.norm(v)


def heldout_r(f, y, groups, pc):
    """Item-grouped CV held-out correlation (probes.fit_one)."""
    preds, _, _ = fit_one(f, y, groups, pc["alpha_grid"], pc["cv_folds"], pc["seed"])
    return float(np.corrcoef(preds, y)[0, 1])


def shuffle_within(y, groups, rng):
    """Permute the target inside each item (label-shuffle null)."""
    out = y.copy()
    for it in np.unique(groups):
        m = groups == it
        out[m] = rng.permutation(out[m])
    return out


def fit_dir(feats_full, y, mask, alpha):
    """Full-set standardization, ridge fit on the masked rows.

    Standardize with mean/std over the full entity set (so the saved mean/std
    match the direction's space, as directions.py does), then fit only on rows
    that have the target. Returns (coef, mean, std) in standardized space.
    """
    mean = feats_full.mean(0)
    std = feats_full.std(0) + 1e-8
    z = (feats_full - mean) / std
    coef = Ridge(alpha).fit(z[mask], y[mask]).coef_
    return coef, mean, std


def gap(feats_full, coef, mean, std, above, below):
    """g: mean(proj | above-median author) - mean(proj | below), unit dir."""
    proj = ((feats_full - mean) / std) @ unit(coef)
    return float(proj[above].mean() - proj[below].mean())


def validate(feats_full, coef, mean, std, target, tmask, n_bins=10):
    """Loading check: proj-vs-target corr + decile monotonicity.

    Returns (proj_r, monotone_bool, bin_means). monotone = projection rises
    with the target across deciles (Pearson of bin index vs bin-mean proj).
    """
    proj = ((feats_full - mean) / std) @ unit(coef)
    p, t = proj[tmask], target[tmask]
    proj_r = float(np.corrcoef(p, t)[0, 1])
    bins = pd.qcut(t, n_bins, labels=False, duplicates="drop")
    means = pd.Series(p).groupby(bins).mean().to_numpy()
    mono_r = float(np.corrcoef(np.arange(len(means)), means)[0, 1])
    return proj_r, mono_r >= 0.9, means


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--judge", default="qwen3-32b",
                    help="recorded judge; its elo file must exist")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: every 4th layer/window, last_token only")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]
    alpha_final = pc["alpha_grid"][-1]

    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    layers, poolings = meta["layers"], meta["poolings"]
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")

    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    assert len(elo) == acts.shape[0], "acts rows do not match filtered elo"
    groups = elo["item"].to_numpy()
    n = len(elo)
    all_mask = np.ones(n, bool)

    # qwen-specific elo covers fewer entities; align by (text_id, item)
    spec = pd.read_csv(out / f"elo_{args.judge}.csv").set_index(["text_id", "item"])["elo"]
    spec_vals = np.array([spec.get((t, i), np.nan)
                          for t, i in zip(elo["text_id"], elo["item"])])
    have = ~np.isnan(spec_vals)

    # capability: author GPQA logit, item-centred (the v_cap target)
    logit = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    author_logit = elo["model"].map(logit).to_numpy(dtype=float)
    cap = author_logit - pd.Series(author_logit).groupby(elo["item"]).transform("mean").to_numpy()

    # alpha unit split: raw author capability, above/below its median
    med = float(np.median(author_logit))
    above, below = author_logit > med, author_logit < med
    print(f"{n} entities, {have.sum()} with {args.judge} elo; "
          f"author-cap median {med:+.3f} -> {above.sum()} above / {below.sum()} below")

    base = {"capability": (cap, all_mask),
            "pooled_elo": (elo["elo"].to_numpy(), all_mask),
            "qwen_elo": (np.nan_to_num(spec_vals), have)}
    # base target -> (single-layer dir name, block dir name, orth name or None)
    dirmap = {"capability": ("v_cap", None),
              "pooled_elo": ("v_pref_gen", "v_pref_gen_orth"),
              "qwen_elo": ("v_pref_spec", "v_pref_spec_orth")}

    feat_cache = {}

    def feat(li, pi):
        k = (li, pi)
        if k not in feat_cache:
            feat_cache[k] = np.asarray(acts[:, li, pi, :], dtype=np.float32)
        return feat_cache[k]

    def block(w, pi):
        return np.concatenate([feat(li, pi) for li in range(w, w + BLOCK)], axis=1)

    layer_idx = range(0, len(layers), 4) if args.quick else range(len(layers))
    n_windows = len(layers) - BLOCK + 1
    win_idx = range(0, n_windows, 4) if args.quick else range(n_windows)
    pool_idx = range(1) if args.quick else range(len(poolings))
    rng = np.random.default_rng(pc["seed"])

    # ---- single-layer sweep -------------------------------------------------
    print("\n== single-layer held-out r ==")
    sl_curve = {t: np.full((len(layers), len(poolings)), np.nan) for t in base}
    for t, (y, mask) in base.items():
        for pi in pool_idx:
            for li in layer_idx:
                sl_curve[t][li, pi] = heldout_r(feat(li, pi)[mask], y[mask],
                                                groups[mask], pc)
        for pi in pool_idx:
            cur = sl_curve[t][:, pi]
            row = "  ".join(f"{layers[li]}:{cur[li]:+.2f}"
                            for li in layer_idx if not np.isnan(cur[li]))
            print(f"{t:11s} {poolings[pi]:15s} {row}")

    # ---- block sweep --------------------------------------------------------
    print("\n== block (5-layer) held-out r ==")
    blk_curve = {t: np.full((n_windows, len(poolings)), np.nan) for t in base}
    for t, (y, mask) in base.items():
        for pi in pool_idx:
            for w in win_idx:
                blk_curve[t][w, pi] = heldout_r(block(w, pi)[mask], y[mask],
                                                groups[mask], pc)
        for pi in pool_idx:
            cur = blk_curve[t][:, pi]
            row = "  ".join(f"{layers[w]}-{layers[w + BLOCK - 1]}:{cur[w]:+.2f}"
                            for w in win_idx if not np.isnan(cur[w]))
            print(f"{t:11s} {poolings[pi]:15s} {row}")

    # ---- build final directions --------------------------------------------
    sl, blk = {}, {}   # name -> saved fields

    def pick(curve):
        i = np.nanargmax(curve)
        return np.unravel_index(i, curve.shape)  # (row, pi)

    for t, (y, mask) in base.items():
        sl_name, orth_name = dirmap[t]
        # single-layer
        li, pi = pick(sl_curve[t])
        f = feat(li, pi)
        coef, mean, std = fit_dir(f, y, mask, alpha_final)
        g = gap(f, coef, mean, std, above, below)
        null = heldout_r(f[mask], shuffle_within(y[mask], groups[mask], rng),
                         groups[mask], pc)
        sl[sl_name] = dict(dir=coef, mean=mean, std=std, layer=layers[li],
                           li=li, pi=pi, pooling=poolings[pi],
                           g=g, r=float(sl_curve[t][li, pi]), null=null)
        # block
        w, pib = pick(blk_curve[t])
        fb = block(w, pib)
        cb, mb, sb = fit_dir(fb, y, mask, alpha_final)
        gb = gap(fb, cb, mb, sb, above, below)
        nullb = heldout_r(block(w, pib)[mask],
                          shuffle_within(y[mask], groups[mask], rng), groups[mask], pc)
        blk[sl_name] = dict(dir=cb.reshape(BLOCK, -1), mean=mb.reshape(BLOCK, -1),
                            std=sb.reshape(BLOCK, -1),
                            layers=[layers[w + j] for j in range(BLOCK)],
                            w=w, pi=pib, pooling=poolings[pib],
                            g=gb, r=float(blk_curve[t][w, pib]), null=nullb)

        # orthogonalized variant: remove v_cap fit at the SAME layer(s)
        if orth_name:
            cap_c, _, _ = fit_dir(f, cap, all_mask, alpha_final)     # v_cap at this layer
            uc = unit(cap_c)
            oc = coef - (uc @ coef) * uc
            sl[orth_name] = dict(dir=oc, mean=mean, std=std, layer=layers[li],
                                 li=li, pi=pi, pooling=poolings[pi],
                                 g=gap(f, oc, mean, std, above, below),
                                 r=sl[sl_name]["r"], null=null)
            cap_cb, _, _ = fit_dir(fb, cap, all_mask, alpha_final)   # v_cap on this window
            ucb = unit(cap_cb)
            ocb = cb - (ucb @ cb) * ucb
            blk[orth_name] = dict(dir=ocb.reshape(BLOCK, -1), mean=mb.reshape(BLOCK, -1),
                                  std=sb.reshape(BLOCK, -1), layers=blk[sl_name]["layers"],
                                  w=w, pi=pib, pooling=poolings[pib],
                                  g=gap(fb, ocb, mb, sb, above, below),
                                  r=blk[sl_name]["r"], null=nullb)

    names5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]

    print("\n== chosen directions ==")
    for name in names5:
        s, b = sl[name], blk[name]
        print(f"{name:17s} single L{s['layer']:>2} {s['pooling']:15s} "
              f"r {s['r']:+.3f} g {s['g']:+.3f}   "
              f"block L{b['layers'][0]}-{b['layers'][-1]} {b['pooling']:15s} "
              f"r {b['r']:+.3f} g {b['g']:+.3f}")

    # ---- validation ---------------------------------------------------------
    # each preference direction validated against its own preference target;
    # v_cap against capability.
    val_target = {"v_cap": (cap, all_mask), "v_pref_gen": (base["pooled_elo"][0], all_mask),
                  "v_pref_gen_orth": (base["pooled_elo"][0], all_mask),
                  "v_pref_spec": (spec_vals, have), "v_pref_spec_orth": (spec_vals, have)}
    print("\n== loading validation (proj vs target) ==")
    valid = {}
    flagged = []
    for name in names5:
        tgt, tm = val_target[name]
        for mode, store in (("single", sl), ("block", blk)):
            d = store[name]
            coef = d["dir"].ravel()
            f = feat(d["li"], d["pi"]) if mode == "single" else block(d["w"], d["pi"])
            pr, mono, _ = validate(f, coef, d["mean"].ravel(), d["std"].ravel(), tgt, tm)
            valid[f"{name}_{mode}"] = (pr, mono)
            if not mono:
                flagged.append(f"{name}_{mode}")
            print(f"{name:17s} {mode:6s} proj-r {pr:+.3f}  monotone {mono}")
    if flagged:
        print("FLAGGED non-monotone:", ", ".join(flagged))

    # ---- legacy flat pod pack (all five at one shared layer) ----------------
    # shared layer = pooled-elo single-layer best (matches old probe_metrics pick)
    fl_li, fl_pi = pick(sl_curve["pooled_elo"])
    ff = feat(fl_li, fl_pi)
    fmean = ff.mean(0)
    fstd = ff.std(0) + 1e-8
    fz = (ff - fmean) / fstd
    flat = {}
    for t, (y, mask) in base.items():
        sl_name, orth_name = dirmap[t]
        flat[sl_name] = Ridge(alpha_final).fit(fz[mask], y[mask]).coef_
    uc = unit(flat["v_cap"])
    for orth_name, src in (("v_pref_gen_orth", "v_pref_gen"),
                           ("v_pref_spec_orth", "v_pref_spec")):
        flat[orth_name] = flat[src] - (uc @ flat[src]) * uc
    dir_norm = float(np.median([np.linalg.norm(flat[k]) for k in names5]))
    rng_r = np.random.default_rng(pc["seed"])
    for r in ("random_a", "random_b"):
        v = rng_r.standard_normal(ff.shape[1])
        flat[r] = v / np.linalg.norm(v) * dir_norm
    flat_names = names5 + ["random_a", "random_b"]
    median_norm = float(np.median(np.linalg.norm(ff, axis=1)))
    print(f"\nflat pod pack: layer {layers[fl_li]} {poolings[fl_pi]} "
          f"median_norm {median_norm:.1f}")

    # ---- self-checks --------------------------------------------------------
    print("\n== self-checks ==")
    assert np.array_equal(base["pooled_elo"][0], elo["elo"].to_numpy()), \
        "(a) v_pref_gen target is not identical to elo.csv elo"
    print("(a) v_pref_gen target == elo.csv elo (raw, not residualized): OK")

    for name in ("v_pref_gen_orth", "v_pref_spec_orth"):
        cap_c, _, _ = fit_dir(feat(sl[name]["li"], sl[name]["pi"]), cap, all_mask, alpha_final)
        cos = float(unit(sl[name]["dir"]) @ unit(cap_c))
        assert abs(cos) < 1e-6, f"(b) {name} not orthogonal to v_cap: cos={cos:.2e}"
        cap_cb, _, _ = fit_dir(block(blk[name]["w"], blk[name]["pi"]), cap, all_mask, alpha_final)
        cosb = float(unit(blk[name]["dir"].ravel()) @ unit(cap_cb))
        assert abs(cosb) < 1e-6, f"(b) {name} block not orthogonal: cos={cosb:.2e}"
    print("(b) all _orth directions orthogonal to v_cap (|cos| < 1e-6): OK")

    assert sl["v_cap"]["g"] > 0, f"(c) v_cap g not positive: {sl['v_cap']['g']}"
    print(f"(c) v_cap g > 0 ({sl['v_cap']['g']:+.3f}): OK")

    cap_peak = np.nanmax(sl_curve["capability"])
    gen_peak = np.nanmax(sl_curve["pooled_elo"])
    print(f"(d) peak held-out r: v_cap {cap_peak:.3f}, v_pref_gen {gen_peak:.3f}")
    assert cap_peak > 0.7, f"(d) v_cap peak r unexpectedly low: {cap_peak:.3f}"
    assert cap_peak > gen_peak, "(d) v_cap should out-predict pooled elo"
    if not (0.50 <= gen_peak <= 0.65):
        print(f"    NOTE: v_pref_gen peak {gen_peak:.3f} outside expected 0.55-0.62 band")

    # ---- save ---------------------------------------------------------------
    save = dict(judge=args.judge, prompt=cfg["record"]["prompt"],
                poolings=np.array(poolings), layers_recorded=np.array(layers),
                block_size=BLOCK,
                # legacy flat pod pack
                layer=layers[fl_li], pooling=poolings[fl_pi],
                mean=fmean, std=fstd, median_norm=median_norm,
                names=np.array(flat_names),
                **{f"dir_{k}": v for k, v in flat.items()},
                **{f"r_{k}": sl[k]["r"] for k in names5})
    # r-curves
    for t in base:
        for pi, pool in enumerate(poolings):
            save[f"slr_{t}_{pool}"] = sl_curve[t][:, pi]
            save[f"blkr_{t}_{pool}"] = blk_curve[t][:, pi]
    # single-layer directions
    for name in names5:
        d = sl[name]
        save[f"sl_{name}_dir"] = d["dir"]
        save[f"sl_{name}_mean"] = d["mean"]
        save[f"sl_{name}_std"] = d["std"]
        save[f"sl_{name}_layer"] = d["layer"]
        save[f"sl_{name}_pooling"] = d["pooling"]
        save[f"sl_{name}_g"] = d["g"]
        save[f"sl_{name}_r"] = d["r"]
        save[f"sl_{name}_null"] = d["null"]
        save[f"sl_{name}_projr"] = valid[f"{name}_single"][0]
        save[f"sl_{name}_monotone"] = valid[f"{name}_single"][1]
    # block directions
    for name in names5:
        d = blk[name]
        save[f"blk_{name}_dir"] = d["dir"]
        save[f"blk_{name}_mean"] = d["mean"]
        save[f"blk_{name}_std"] = d["std"]
        save[f"blk_{name}_layers"] = np.array(d["layers"])
        save[f"blk_{name}_pooling"] = d["pooling"]
        save[f"blk_{name}_g"] = d["g"]
        save[f"blk_{name}_r"] = d["r"]
        save[f"blk_{name}_null"] = d["null"]
        save[f"blk_{name}_projr"] = valid[f"{name}_block"][0]
        save[f"blk_{name}_monotone"] = valid[f"{name}_block"][1]

    path = out / ("directions_multilayer_quick.npz" if args.quick
                  else "directions_multilayer.npz")
    np.savez(path, **save)
    print(f"\n-> {path}  ({len(save)} keys)")


if __name__ == "__main__":
    main()
