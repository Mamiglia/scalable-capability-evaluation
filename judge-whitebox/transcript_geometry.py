"""Strength-axis geometry on single-transcript recordings (record_resid.py --single-transcript):
per layer, does a "how capable is this author" direction exist within an item's transcripts, and
does it (or its distance structure) transfer between two models' recordings of the same texts.

strength_axis: within-item strong/weak author classifier (CCGP-style: item-grouped CV so no item
leaks between train/test) + RSA-lite (representational distance vs |cap gap|), on ONE recording.
stratified_cka: linear CKA between TWO recordings' residual streams over the same transcripts
(aligned by text hash, not row position -- the two models may tokenize/truncate differently),
split into all/strong/weak strata.
"""
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import resid_probes
import wbconfig
import wbfig

MIN_TRANSCRIPTS_PER_ITEM = 2  # can't rank strong-vs-weak within an item from a single transcript


def _assert_single_transcript(rec, label):
    assert rec.get("mode") == "single_transcript", (
        f"{label} must be a single_transcript recording (record_resid.py --single-transcript); "
        f"got mode={rec.get('mode')!r}")


def _strong_weak(item, cap):
    """Rank transcripts within each item by cap: upper half -> 1 (strong), lower half -> 0 (weak).
    Odd-sized items put the extra (middle-ranked) transcript in the strong half. Items with fewer
    than 2 distinct cap values can't be split and are excluded (keep=False for all their rows).
    Returns (label[n] int, keep[n] bool)."""
    item, cap = np.asarray(item), np.asarray(cap, dtype=float)
    label, keep = np.full(len(item), -1, dtype=int), np.zeros(len(item), dtype=bool)
    for it in pd.unique(item):
        idx = np.flatnonzero(item == it)
        if len(set(cap[idx])) < 2:
            continue
        order = idx[np.argsort(cap[idx], kind="stable")]  # ascending cap
        half = len(idx) // 2
        label[order[:half]], label[order[half:]] = 0, 1
        keep[idx] = True
    return label, keep


def _center_within_item(X, item):
    """Subtract each item's own mean representation from its transcripts -- removes item/content
    variance so the strong/weak classifier reads author strength, not which question it is."""
    Xc = X.copy()
    for it in pd.unique(item):
        mask = item == it
        Xc[mask] -= X[mask].mean(axis=0, keepdims=True)
    return Xc


def _rsa_lite(X, item, cap, rsa_min_items):
    """Pooled Spearman corr between within-item pairwise Euclidean distance in X and |cap gap|,
    across items with >=2 transcripts (pairs, not just the strong/weak split -- ties in cap still
    contribute a zero gap). Returns (r, p, n_items); (nan, nan, n_items) if fewer than
    rsa_min_items items qualify (too few pairs for a meaningful correlation)."""
    dists, gaps, n_items = [], [], 0
    for it in pd.unique(item):
        idx = np.flatnonzero(item == it)
        if len(idx) < MIN_TRANSCRIPTS_PER_ITEM:
            continue
        n_items += 1
        for a, b in itertools.combinations(idx, 2):
            dists.append(float(np.linalg.norm(X[a] - X[b])))
            gaps.append(float(abs(cap[a] - cap[b])))
    if n_items < rsa_min_items:
        return float("nan"), float("nan"), n_items
    r, p = spearmanr(dists, gaps)
    return float(r), float(p), n_items


def strength_axis(rec, alphas, cv_folds, rsa_min_items):
    """Per layer: (a) strong-vs-weak author classifier after within-item centring, item-grouped CV
    (resid_probes._probe_layer -- same ridge/CV machinery as the judging-pair probes) -> held-out
    accuracy + direction; (b) RSA-lite. Returns (summary df[layer, alpha, accuracy, rsa_r, rsa_p,
    n_items], directions[n_layers, d])."""
    _assert_single_transcript(rec, "rec")
    item, cap = np.asarray(rec["item"]), np.asarray(rec["cap"], dtype=float)
    label, keep = _strong_weak(item, cap)
    assert keep.sum() > 0, "no items have >=2 distinct cap values -- can't build a strong/weak split"
    resid = np.asarray(rec["resid"])  # [n, n_layers, d]

    rows, directions = [], []
    for layer in range(resid.shape[1]):
        X = _center_within_item(resid[:, layer, :], item)
        alpha, acc, direction, _ = resid_probes._probe_layer(
            X[keep], label[keep], item[keep], alphas, cv_folds)
        rsa_r, rsa_p, n_items = _rsa_lite(resid[:, layer, :], item, cap, rsa_min_items)
        rows.append({"layer": layer, "alpha": alpha, "accuracy": acc,
                      "rsa_r": rsa_r, "rsa_p": rsa_p, "n_items": n_items})
        directions.append(direction)
    return pd.DataFrame(rows), np.stack(directions)


def _align_by_hash(rec_a, rec_b):
    """Positions into rec_a/rec_b's resid tensors for transcripts BOTH kept, matched by text hash
    (not row position -- the two recordings may tokenize/truncate the same texts differently, so
    kept_unit_idx from record_resid.py isn't a shared index the way pairwise kept_row_idx is;
    extends resid_probes._align_rows' role for this recording shape)."""
    pos_a = {h: i for i, h in enumerate(rec_a["text_hash"])}
    pos_b = {h: i for i, h in enumerate(rec_b["text_hash"])}
    common = sorted(set(pos_a) & set(pos_b))
    return [pos_a[h] for h in common], [pos_b[h] for h in common]


def stratified_cka(rec_a, rec_b):
    """Per layer, linear CKA between rec_a/rec_b's residual streams over shared transcripts
    (aligned by text hash), computed three ways: all shared transcripts, the strong-author half,
    the weak-author half (strata from rec_a's own item/cap -- same transcript, same cap, on both
    sides by construction, so rec_b's labels would agree)."""
    _assert_single_transcript(rec_a, "rec_a")
    _assert_single_transcript(rec_b, "rec_b")
    resid_probes._assert_pairs_path(rec_b.get("pairs_path"), rec_a.get("pairs_path"),
                                     "rec_b", ref_label="rec_a")

    idx_a, idx_b = _align_by_hash(rec_a, rec_b)
    item_a = np.asarray(rec_a["item"])[idx_a]
    cap_a = np.asarray(rec_a["cap"], dtype=float)[idx_a]
    label, keep = _strong_weak(item_a, cap_a)

    resid_a = np.asarray(rec_a["resid"])[idx_a]
    resid_b = np.asarray(rec_b["resid"])[idx_b]
    n_layers = min(resid_a.shape[1], resid_b.shape[1])
    rows = []
    for layer in range(n_layers):
        Xa, Xb = resid_a[:, layer, :], resid_b[:, layer, :]
        row = {"layer": layer, "cka_all": resid_probes.linear_cka(Xa, Xb)}
        for name, val in (("strong", 1), ("weak", 0)):
            mask = keep & (label == val)
            # ponytail: no min-N guard here -- a near-empty stratum falls through to linear_cka's
            # own den>0 check (returns 0.0, not nan). Fine at real recording scale; add a min-N
            # guard (mirroring rsa_min_items) if strata get small enough for that to mislead.
            row[f"cka_{name}"] = resid_probes.linear_cka(Xa[mask], Xb[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def fig_strength_accuracy(probes, tag):
    """Held-out strong-vs-weak accuracy vs layer, one line per recording (base always; ft
    alongside if given)."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    for i, (name, (summary, _)) in enumerate(probes.items()):
        ax.plot(summary["layer"], summary["accuracy"], label=name, color=wbfig.PALETTE[i % len(wbfig.PALETTE)])
    ax.axhline(0.5, color=wbfig.GRID, linewidth=1, zorder=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("strong-vs-weak held-out accuracy")
    ax.legend(frameon=False)
    wbfig.save(fig, f"transcript_geometry_{tag}_accuracy")


def fig_rsa(probes, tag):
    """RSA-lite correlation vs layer, one line per recording."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    for i, (name, (summary, _)) in enumerate(probes.items()):
        ax.plot(summary["layer"], summary["rsa_r"], label=name, color=wbfig.PALETTE[i % len(wbfig.PALETTE)])
    ax.axhline(0.0, color=wbfig.GRID, linewidth=1, zorder=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("RSA: dist vs |cap gap| (Spearman r)")
    ax.legend(frameon=False)
    wbfig.save(fig, f"transcript_geometry_{tag}_rsa")


def fig_cka(cka_df, tag):
    """Linear CKA vs layer, three lines: all shared transcripts, strong half, weak half."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    for i, name in enumerate(("all", "strong", "weak")):
        ax.plot(cka_df["layer"], cka_df[f"cka_{name}"], label=name, color=wbfig.PALETTE[i])
    ax.set_xlabel("layer")
    ax.set_ylabel("linear CKA, rec A vs rec B")
    ax.legend(frameon=False)
    wbfig.save(fig, f"transcript_geometry_{tag}_cka")


def _strength_csv(probes, out_dir, tag):
    rows = [{"model": name, "layer": r.layer, "alpha": r.alpha, "accuracy": r.accuracy,
             "rsa_r": r.rsa_r, "rsa_p": r.rsa_p, "n_items": r.n_items}
            for name, (summary, _) in probes.items() for r in summary.itertuples()]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"transcript_geometry_{tag}_strength.csv", index=False)
    return df


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-a", required=True, help="base single-transcript recording")
    parser.add_argument("--rec-b", default=None, help="optional second (e.g. FT) recording")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    rec_a = torch.load(args.rec_a)
    rec_b = torch.load(args.rec_b) if args.rec_b else None

    alphas, cv_folds = c["resid"]["alphas"], c["resid"]["cv_folds"]
    rsa_min_items = c["transcript_geometry"]["rsa_min_items"]

    probes = {"a": strength_axis(rec_a, alphas, cv_folds, rsa_min_items)}
    if rec_b is not None:
        probes["b"] = strength_axis(rec_b, alphas, cv_folds, rsa_min_items)

    out_dir = pathlib.Path(c["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"probes": probes, "tag": args.tag}, out_dir / f"transcript_geometry_{args.tag}.pt")
    _strength_csv(probes, out_dir, args.tag)
    fig_strength_accuracy(probes, args.tag)
    fig_rsa(probes, args.tag)

    cka_df = None
    if rec_b is not None:
        cka_df = stratified_cka(rec_a, rec_b)
        cka_df.to_csv(out_dir / f"transcript_geometry_{args.tag}_cka.csv", index=False)
        fig_cka(cka_df, args.tag)

    print(f"models={list(probes)} cka={'yes' if cka_df is not None else 'no'} -> {out_dir}")


if __name__ == "__main__":
    main()
