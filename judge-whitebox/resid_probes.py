"""Decision-subspace probes on recorded residual streams (record_resid.py output): per-layer
linear probes for the model's own verdict, the ground-truth winner, and (given both instruction
polarities) which prompt asked. Base vs FT comparison: principal angle between probe directions
and linear CKA between residual-stream matrices, both per layer.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold

import wbconfig
import wbfig
import wbpairs

TARGETS = ("verdict", "winner", "polarity")


def _assert_pairs_path(rec_path, ref_path, label, ref_label="reference"):
    """Two files can share a split NAME (e.g. an "eval_mm" split exists in both the main pairs
    file and a sweep file), so split/limit equality alone doesn't prove kept_row_idx indexes into
    the same parquet -- pairs_path is the field that actually pins that down. Missing on either
    side (legacy recording, pre this field) is a WARNING, not a silent pass-through."""
    if rec_path is None or ref_path is None:
        print(f"WARNING: legacy recording missing pairs_path, skipping pairs-file identity check: {label}")
        return
    assert rec_path == ref_path, (
        f"pairs-file mismatch at {label}: pairs_path={rec_path!r} != {ref_label} pairs_path={ref_path!r}")


def _assert_aligned(rec, ref, label):
    """rec's split/limit/pairs_path must match ref's -- every consumer that reuses ONE pairs_df or
    lines up kept_row_idx across two recordings (fit_probes' polarity target, compare_models' CKA,
    main's shared pairs_df) silently mislabels ground truth or pairs unrelated prompts otherwise.
    label names the mismatched recording in the message: a file path when known, else a role name."""
    assert rec["split"] == ref["split"] and rec["limit"] == ref["limit"], (
        f"recording alignment mismatch at {label}: split={rec['split']!r} limit={rec['limit']!r} "
        f"!= reference split={ref['split']!r} limit={ref['limit']!r}")
    _assert_pairs_path(rec.get("pairs_path"), ref.get("pairs_path"), label)


def _kept_meta(rec, pairs_df):
    """pairs_df rows aligned to rec's kept prompts, via record_resid's own kept_row_idx (same
    positional-alignment convention as ScreenResult -- reuse wbpairs.select_kept_rows)."""
    screen_like = {"set_name": "judging", "n_prompts": rec["resid"].shape[0],
                   "kept_row_idx": rec["kept_row_idx"]}
    return wbpairs.select_kept_rows(pairs_df, screen_like)


def _probe_layer(X, y, groups, alphas, cv_folds):
    """Ridge classifier, alpha chosen by item-grouped CV (max mean held-out accuracy); direction
    refit on all data at that alpha. Returns (best_alpha, cv_accuracy, direction[d], intercept)."""
    n_splits = min(cv_folds, len(set(groups)))
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    best_alpha, best_acc = alphas[0], -1.0
    for alpha in alphas:
        accs = [RidgeClassifier(alpha=alpha).fit(X[tr], y[tr]).score(X[te], y[te]) for tr, te in splits]
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_alpha, best_acc = alpha, acc
    clf = RidgeClassifier(alpha=best_alpha).fit(X, y)
    direction = np.ravel(clf.coef_).astype(np.float32)
    return best_alpha, best_acc, direction, float(np.ravel(clf.intercept_)[0])


def _fit_per_layer(resid, y, groups, alphas, cv_folds):
    """resid: [n, n_layers, d] (tensor or array). Returns (summary df: layer/alpha/accuracy,
    directions: float32 [n_layers, d])."""
    resid = np.asarray(resid)
    rows, directions = [], []
    for layer in range(resid.shape[1]):
        alpha, acc, direction, intercept = _probe_layer(resid[:, layer, :], y, groups, alphas, cv_folds)
        rows.append({"layer": layer, "alpha": alpha, "accuracy": acc, "intercept": intercept})
        directions.append(direction)
    return pd.DataFrame(rows), np.stack(directions)


def fit_probes(rec, pairs_df, targets, alphas, cv_folds, rec_inv=None):
    """rec/rec_inv: record_resid.py save-dicts for the SAME model+adapter (rec_inv = the
    inverted-instruction recording, needed only for the "polarity" target). pairs_df:
    wbpairs.load_pairs output covering rec's split/limit. Returns {target: (summary_df, directions)}.
    """
    meta = _kept_meta(rec, pairs_df)
    out = {}
    if "verdict" in targets:
        y = (rec["l1"] > rec["l2"]).numpy().astype(int)
        out["verdict"] = _fit_per_layer(rec["resid"], y, meta["item"].to_numpy(), alphas, cv_folds)
    if "winner" in targets:
        y = (meta["label"].to_numpy() == 1).astype(int)
        out["winner"] = _fit_per_layer(rec["resid"], y, meta["item"].to_numpy(), alphas, cv_folds)
    if "polarity" in targets:
        assert rec_inv is not None, "polarity target needs rec_inv"
        assert rec["model"] == rec_inv["model"] and rec["adapter"] == rec_inv["adapter"], (
            "polarity probe compares normal vs inverted prompts of the SAME model+adapter")
        _assert_aligned(rec_inv, rec, "rec_inv")
        meta_inv = _kept_meta(rec_inv, pairs_df)
        X = torch.cat([rec["resid"], rec_inv["resid"]], dim=0)
        y = np.concatenate([np.zeros(rec["resid"].shape[0]), np.ones(rec_inv["resid"].shape[0])]).astype(int)
        groups = np.concatenate([meta["item"].to_numpy(), meta_inv["item"].to_numpy()])
        out["polarity"] = _fit_per_layer(X, y, groups, alphas, cv_folds)
    return out


def principal_angle_deg(u, v):
    """Unsigned angle in degrees between two probe directions (sign is probe-arbitrary)."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    cos = 0.0 if denom == 0 else abs(np.dot(u, v) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def linear_cka(X, Y):
    """Biased linear CKA (Kornblith et al. 2019), plain formula: centre columns of same-n
    [n, d1]/[n, d2] representations, CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)."""
    Xc, Yc = X - X.mean(axis=0, keepdims=True), Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(num / den) if den > 0 else 0.0


def _align_rows(rec_a, rec_b):
    """Positions into rec_a/rec_b's resid tensors for the prompts BOTH kept (same original
    pairs_df row, via kept_row_idx) -- CKA needs the two matrices to represent the same inputs."""
    pos_a = {r: i for i, r in enumerate(rec_a["kept_row_idx"])}
    pos_b = {r: i for i, r in enumerate(rec_b["kept_row_idx"])}
    common = sorted(set(pos_a) & set(pos_b))
    return [pos_a[r] for r in common], [pos_b[r] for r in common]


def compare_models(base_probes, ft_probes, rec_base, rec_ft):
    """Per shared target, per layer: principal angle between base/FT probe directions.
    Per layer: linear CKA between base/FT resid matrices (rows aligned to shared prompts)."""
    _assert_aligned(rec_ft, rec_base, "rec_b_norm")
    angle_rows = []
    for target in sorted(base_probes.keys() & ft_probes.keys()):
        _, dirs_a = base_probes[target]
        _, dirs_b = ft_probes[target]
        n_layers = min(dirs_a.shape[0], dirs_b.shape[0])
        for layer in range(n_layers):
            angle_rows.append({"target": target, "layer": layer,
                                "angle_deg": principal_angle_deg(dirs_a[layer], dirs_b[layer])})
    angle_df = pd.DataFrame(angle_rows)

    idx_a, idx_b = _align_rows(rec_base, rec_ft)
    n_layers = min(rec_base["resid"].shape[1], rec_ft["resid"].shape[1])
    cka_rows = [{"layer": layer,
                 "cka": linear_cka(rec_base["resid"][idx_a, layer, :].numpy(),
                                    rec_ft["resid"][idx_b, layer, :].numpy())}
                for layer in range(n_layers)]
    return angle_df, pd.DataFrame(cka_rows)


def fig_accuracy(base_probes, ft_probes, tag):
    """Held-out accuracy vs layer, one line per target, one panel per model (base always; FT
    alongside if given). Chance (0.5) marked; every target has a legend entry."""
    models = [("base", base_probes)] + ([("ft", ft_probes)] if ft_probes else [])
    fig = wbfig.new_fig(wbfig.WIDE, 3.0)
    for panel, (name, probes) in enumerate(models):
        ax = fig.add_subplot(1, len(models), panel + 1)
        for i, target in enumerate(TARGETS):
            if target in probes:
                summary, _ = probes[target]
                ax.plot(summary["layer"], summary["accuracy"], label=target,
                        color=wbfig.PALETTE[i % len(wbfig.PALETTE)])
        ax.axhline(0.5, color=wbfig.GRID, linewidth=1, zorder=0)
        ax.set_xlabel(f"layer ({name})")
        if panel == 0:
            ax.set_ylabel("held-out accuracy")
        ax.legend(frameon=False)
    wbfig.save(fig, f"resid_probes_{tag}_accuracy")


def fig_principal_angle(angle_df, tag):
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    for i, target in enumerate(TARGETS):
        sub = angle_df[angle_df["target"] == target]
        if len(sub):
            ax.plot(sub["layer"], sub["angle_deg"], label=target, color=wbfig.PALETTE[i % len(wbfig.PALETTE)])
    ax.axhline(90, color=wbfig.GRID, linewidth=1, zorder=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("principal angle, base vs FT (deg)")
    ax.legend(frameon=False)
    wbfig.save(fig, f"resid_probes_{tag}_angle")


def fig_cka(cka_df, tag):
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    ax.plot(cka_df["layer"], cka_df["cka"], color=wbfig.PALETTE[0])
    ax.set_xlabel("layer")
    ax.set_ylabel("linear CKA, base vs FT resid")
    wbfig.save(fig, f"resid_probes_{tag}_cka")


def _summary_csv(base_probes, ft_probes, out_dir, tag):
    rows = []
    for model_name, probes in (("base", base_probes), ("ft", ft_probes or {})):
        for target, (summary, _) in probes.items():
            for r in summary.itertuples():
                rows.append({"model": model_name, "target": target, "layer": r.layer,
                             "alpha": r.alpha, "accuracy": r.accuracy})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"resid_probes_{tag}.csv", index=False)
    return df


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-norm", required=True)
    parser.add_argument("--rec-inv", default=None)
    parser.add_argument("--rec-b-norm", default=None)
    parser.add_argument("--rec-b-inv", default=None)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    rec_norm = torch.load(args.rec_norm)
    rec_inv = torch.load(args.rec_inv) if args.rec_inv else None
    rec_b_norm = torch.load(args.rec_b_norm) if args.rec_b_norm else None
    rec_b_inv = torch.load(args.rec_b_inv) if args.rec_b_inv else None

    # pairs_df below is loaded ONCE, keyed off rec_norm's split/limit, and every recording's
    # kept_row_idx gets decoded positionally against it -- a split/limit/pairs_path mismatch on
    # any of them would silently mislabel ground truth or pair up unrelated prompts, so check
    # before any of that reuse happens. rec_norm itself is checked against what THIS run's cfg
    # resolves to -- pairs_df is about to be loaded from that path, not from rec_norm's recorded one.
    _assert_pairs_path(rec_norm.get("pairs_path"), c["paths"]["pairs"], args.rec_norm, ref_label="current cfg")
    for rec, path in ((rec_inv, args.rec_inv), (rec_b_norm, args.rec_b_norm), (rec_b_inv, args.rec_b_inv)):
        if rec is not None:
            _assert_aligned(rec, rec_norm, path)

    pairs_df = wbpairs.load_pairs(c, split=rec_norm["split"], limit=rec_norm["limit"])
    alphas, cv_folds = c["resid"]["alphas"], c["resid"]["cv_folds"]

    targets = ["verdict", "winner"] + (["polarity"] if rec_inv is not None else [])
    base_probes = fit_probes(rec_norm, pairs_df, targets, alphas, cv_folds, rec_inv=rec_inv)

    ft_probes, angle_df, cka_df = None, None, None
    if rec_b_norm is not None:
        targets_b = ["verdict", "winner"] + (["polarity"] if rec_b_inv is not None else [])
        ft_probes = fit_probes(rec_b_norm, pairs_df, targets_b, alphas, cv_folds, rec_inv=rec_b_inv)
        angle_df, cka_df = compare_models(base_probes, ft_probes, rec_norm, rec_b_norm)

    out_dir = pathlib.Path(c["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"base": base_probes, "ft": ft_probes, "angle": angle_df, "cka": cka_df,
                "tag": args.tag}, out_dir / f"resid_probes_{args.tag}.pt")
    _summary_csv(base_probes, ft_probes, out_dir, args.tag)

    fig_accuracy(base_probes, ft_probes, args.tag)
    if angle_df is not None:
        fig_principal_angle(angle_df, args.tag)
        fig_cka(cka_df, args.tag)

    print(f"targets={list(base_probes)} ft={'yes' if ft_probes else 'no'} -> {out_dir}")


if __name__ == "__main__":
    main()
