"""Order/polarity component classification: order-symmetric vs order-anti-symmetric attribution
mass (slot-anchored vs transcript-anchored, per pair_uid/order pairing), and behaviour under
instruction-polarity inversion (assessment / decision-state / bypass / mixed), per component.

Order pairing: anti-symmetric attribution across order 0/1 = transcript-anchored (tracks content,
which flips slot relative to the fixed slot-based loss); symmetric = slot-anchored (a content-
independent position bias). Polarity classification (normal vs inverted verdict prompt, same
model+adapter so components share a decomp basis): activations (per_prompt_proj) stable but
attribution (per_prompt_attr) flips sign = an assessment feature upstream of a working inversion;
activations flip = decision-state; neither flips = an instruction-bypassing channel (bias).

order_regression/affine_stats/reclassify (below) are the primary statistics -- coherence and the
Pearson-r class (polarity_compare) are offset-blind: y = x + constant still correlates at r=1, so
a component with a real instruction-driven mean shift reads as order-symmetric "bypass" under the
old classifier. order_regression replaces coherence as the primary order statistic (paired
slot/winner-side regression, pair-clustered SEs); affine_stats/reclassify replace the Pearson-r
polarity class with a mean-shift/scale/effect decomposition, bootstrap CI, and a minimum-effect
gate, defaulting unidentified components to "unidentified" rather than a false "bypass". Both old
outputs are kept as legacy columns for comparison.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

import before_after
import importance_report
import svd_components
import wbconfig
import wbfig
import wbpairs

CLASSES = ("assessment", "decision-state", "bypass", "mixed")
CLASS_COLORS = {"assessment": wbfig.PALETTE[0], "decision-state": wbfig.PALETTE[1],
                 "bypass": wbfig.PALETTE[2], "mixed": wbfig.MUTED}

BOOL_LABELS = ("instruction_sensitive", "slot", "content", "unidentified")
BOOL_COLORS = {lbl: wbfig.PALETTE[i] for i, lbl in enumerate(BOOL_LABELS)}


def _assert_no_dup_pair_order(meta):
    assert not meta.duplicated(["pair_uid", "order"]).any(), \
        "duplicate pair_uid/order rows in screen (one row per order per pair expected)"


def order_split(screen: dict, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Pair per_prompt_attr rows by pair_uid across order 0/1; per component j: sym[j] = mean
    over pairs of (a0+a1)/2, anti[j] = mean over pairs of (a0-a1)/2, plus mean |sym|/|anti|
    magnitudes (abs taken per-pair, before averaging). Pairs missing either order (dropped by
    screen.py's token-length cap) are excluded and counted in n_dropped_pairs. Requires
    per_prompt_attr (task 14+ screens) -- see coherence() for a legacy-compatible alternative.
    """
    attr = screen.get("per_prompt_attr")
    assert attr is not None, "order_split requires per_prompt_attr (re-run screen.py, task 14+)"
    attr = np.asarray(attr)
    top_k = screen["top_k_components"]
    meta = wbpairs.select_kept_rows(pairs_df, screen)
    _assert_no_dup_pair_order(meta)
    assert len(meta) == attr.shape[0], "prompt_meta / per_prompt_attr row count mismatch"

    wide = pd.DataFrame(attr, columns=range(attr.shape[1]))
    wide["pair_uid"], wide["order"] = meta["pair_uid"].to_numpy(), meta["order"].to_numpy()
    cols = list(range(attr.shape[1]))

    piv0 = wide[wide["order"] == 0].set_index("pair_uid")
    piv1 = wide[wide["order"] == 1].set_index("pair_uid")
    shared = piv0.index.intersection(piv1.index)
    n_dropped = wide["pair_uid"].nunique() - len(shared)

    a0, a1 = piv0.loc[shared, cols].to_numpy(), piv1.loc[shared, cols].to_numpy()
    sym, anti = (a0 + a1) / 2, (a0 - a1) / 2

    rows = []
    for j, (key, comp) in enumerate(top_k):
        rows.append({
            "key": key, "comp": comp,
            "sym": sym[:, j].mean(), "anti": anti[:, j].mean(),
            "abs_sym": np.abs(sym[:, j]).mean(), "abs_anti": np.abs(anti[:, j]).mean(),
            "n_pairs": len(shared), "n_dropped_pairs": int(n_dropped),
        })
    return pd.DataFrame(rows)


def coherence(screen: dict) -> pd.DataFrame:
    """Order-coherence ratio |attr_signed| / attr_abs per component, from the screen's aggregate
    (across-all-prompts) fields -- works on any screen (judging or control, legacy or not), since
    it needs neither per_prompt_attr nor pairs_df. attr_abs > 0 for every stored top_k component
    (screen.py only stores components with nonzero activity), so no zero-denominator guard needed.
    """
    attr_signed, attr_abs = screen["attr_signed"], screen["attr_abs"]
    rows = []
    for key, comp in screen["top_k_components"]:
        s, a = abs(float(attr_signed[key][comp])), float(attr_abs[key][comp])
        rows.append({"key": key, "comp": comp, "coherence": s / a if a else float("nan")})
    return pd.DataFrame(rows)


def _corr(a, b):
    """Pearson r, nan if either side has no variance (constant array -- pearsonr divides by 0)."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _boot_pair_idx(pair_uid, boot_n: int, seed: int) -> list:
    """Precompute boot_n item(pair)-grouped resample row-index arrays once -- reused across every
    component's bootstrap CI instead of re-deriving the pair grouping per component. Resamples
    pair_uids with replacement and keeps all rows of each resampled pair together, so the two
    order rows of one pair move as a unit (not independently -- they're not independent
    observations)."""
    pair_uid = np.asarray(pair_uid)
    uids, inverse = np.unique(pair_uid, return_inverse=True)
    groups = [np.where(inverse == g)[0] for g in range(len(uids))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uids), size=(boot_n, len(uids)))
    return [np.concatenate([groups[g] for g in draw]) for draw in draws]


def affine_stats(x, y, boot_idx=None) -> dict:
    """Paired affine decomposition of y relative to x on identical rows (e.g. attribution, normal
    vs inverted instruction polarity): mean_shift = mean(y-x); scale = std(y)/std(x) (nan if x is
    constant); r_centred = Pearson r on centred values (equal to plain Pearson r -- centring
    doesn't change it, kept explicit to match the review's formula); effect = mean_shift / pooled
    std (nan if both sides are constant). Pearson r alone is offset-blind: y = x + constant still
    correlates at r=1, which is exactly the failure this replaces (a real instruction-driven mean
    offset was reading as order-symmetric "bypass" under the old r-only classifier). boot_idx:
    precomputed item(pair)-grouped resample row-index arrays from _boot_pair_idx -- ci_lo/ci_hi
    are nan if not given."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diff = y - x
    mean_shift = float(diff.mean())
    std_x, std_y = float(x.std()), float(y.std())
    scale = std_y / std_x if std_x > 0 else float("nan")
    pooled_std = float(np.sqrt((std_x ** 2 + std_y ** 2) / 2))
    effect = mean_shift / pooled_std if pooled_std > 0 else float("nan")

    ci_lo, ci_hi = float("nan"), float("nan")
    if boot_idx:
        boot_means = np.array([diff[idx].mean() for idx in boot_idx])
        ci_lo, ci_hi = (float(v) for v in np.percentile(boot_means, [2.5, 97.5]))

    return {"mean_shift": mean_shift, "scale": scale, "r_centred": _corr(x, y),
            "effect": effect, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS beta + cluster-robust (CR1, small-sample corrected) SE, clustered on `groups`. pinv
    (not inv) so a rank-deficient design -- e.g. a degenerate all-one-winner-side test fixture --
    returns a least-norm fit instead of raising."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    uniq = np.unique(groups)
    for g in uniq:
        m = groups == g
        score = X[m].T @ resid[m]
        meat += np.outer(score, score)
    n, k, n_groups = X.shape[0], X.shape[1], len(uniq)
    correction = (n_groups / (n_groups - 1)) * ((n - 1) / (n - k)) if n_groups > 1 and n > k else 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se


def order_regression(screen: dict, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Paired regression a ~ slot + winner_side + slot:winner_side per component -- the primary
    order statistic (order_split's sym/anti and coherence stay as legacy columns, see main()).
    slot = order coded +-1 (order 0 = +1). winner_side = +1 iff the winning transcript sits in
    slot 1 *for this specific row* (raw label == 1), row-wise -- this FLIPS across the two order
    rows of a pair (an order swap swaps text_1/text_2, so the winner's slot number necessarily
    flips too), it is intentionally not pair-invariant. (A pair-invariant "which content is the
    winner" identity is exactly slot*winner_side -- that's the interaction term, not winner_side
    itself; conflating the two swaps beta_winner and beta_inter, which a prior version of this
    function did.) OLS per component, pair-clustered SEs (_ols_cluster)."""
    attr = np.asarray(screen["per_prompt_attr"])
    meta = wbpairs.select_kept_rows(pairs_df, screen)
    order = meta["order"].to_numpy()
    label = meta["label"].to_numpy()
    slot = np.where(order == 0, 1.0, -1.0)
    winner = np.where(label == 1, 1.0, -1.0)
    X = np.column_stack([np.ones_like(slot), slot, winner, slot * winner])
    groups = meta["pair_uid"].to_numpy()

    rows = []
    for j, (key, comp) in enumerate(screen["top_k_components"]):
        beta, se = _ols_cluster(X, attr[:, j], groups)
        rows.append({
            "key": key, "comp": comp,
            "beta_intercept": beta[0], "beta_slot": beta[1],
            "beta_winner": beta[2], "beta_inter": beta[3],
            "se_intercept": se[0], "se_slot": se[1],
            "se_winner": se[2], "se_inter": se[3],
        })
    return pd.DataFrame(rows)


def _classify(attr_r, proj_r, thresh):
    """assessment: proj stable+positive (activations unchanged) with attr flipped (attribution
    inverts) -- an assessment feature upstream of a working inversion. decision-state: proj
    flipped -- the activation itself inverts. bypass: BOTH proj and attr stable+positive --
    instruction ignored outright. else (including either r merely ambiguous, |r| < thresh, or
    nan): mixed -- not a clean fit for any of the above."""
    if np.isnan(attr_r) or np.isnan(proj_r):
        return "mixed"
    proj_stable, proj_flipped = proj_r >= thresh, proj_r <= -thresh
    attr_stable, attr_flipped = attr_r >= thresh, attr_r <= -thresh
    if proj_stable and attr_flipped:
        return "assessment"
    if proj_flipped:
        return "decision-state"
    if proj_stable and attr_stable:
        return "bypass"
    return "mixed"


def _align_polarity(screen_norm: dict, screen_inv: dict, pairs_df: pd.DataFrame):
    """Shared (pair_uid, order) row alignment + shared component list between two polarity
    screens (normal vs inverted instruction) -- common setup for both the legacy Pearson-r
    classifier (polarity_compare) and the affine stats (polarity_affine). Same model+adapter
    required (same decomp basis, so (key, comp) indexes the same direction on both sides)."""
    assert screen_norm.get("model") == screen_inv.get("model") and \
        screen_norm.get("adapter") == screen_inv.get("adapter"), (
            f"requires the same model+adapter on both sides: "
            f"norm={screen_norm.get('model')!r}/{screen_norm.get('adapter')!r} vs "
            f"inv={screen_inv.get('model')!r}/{screen_inv.get('adapter')!r}")
    for s in (screen_norm, screen_inv):
        assert s.get("per_prompt_attr") is not None and s.get("per_prompt_proj") is not None, \
            "requires per_prompt_attr/per_prompt_proj (task 14+ screens)"

    meta_norm = wbpairs.select_kept_rows(pairs_df, screen_norm)
    meta_inv = wbpairs.select_kept_rows(pairs_df, screen_inv)
    _assert_no_dup_pair_order(meta_norm)
    _assert_no_dup_pair_order(meta_inv)
    idx_norm = {k: i for i, k in enumerate(zip(meta_norm["pair_uid"], meta_norm["order"]))}
    idx_inv = {k: i for i, k in enumerate(zip(meta_inv["pair_uid"], meta_inv["order"]))}
    shared_keys = sorted(set(idx_norm) & set(idx_inv))
    rows_norm = [idx_norm[k] for k in shared_keys]
    rows_inv = [idx_inv[k] for k in shared_keys]
    pair_uid = np.array([k[0] for k in shared_keys])

    comps_norm = {c: j for j, c in enumerate(screen_norm["top_k_components"])}
    comps_inv = {c: j for j, c in enumerate(screen_inv["top_k_components"])}
    shared_comps = [c for c in comps_norm if c in comps_inv]

    attr_norm = np.asarray(screen_norm["per_prompt_attr"])[rows_norm]
    attr_inv = np.asarray(screen_inv["per_prompt_attr"])[rows_inv]
    proj_norm = np.asarray(screen_norm["per_prompt_proj"])[rows_norm]
    proj_inv = np.asarray(screen_inv["per_prompt_proj"])[rows_inv]
    return shared_comps, comps_norm, comps_inv, attr_norm, attr_inv, proj_norm, proj_inv, pair_uid


def polarity_compare(screen_norm: dict, screen_inv: dict, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy polarity class (kept for comparison -- see reclassify() for the primary
    classification). Per component shared by both screens: attr_flip_rate/attr_r = sign-flip rate
    and Pearson r of per_prompt_attr across polarity; proj_flip_rate/proj_r = same for
    per_prompt_proj. class per _classify (config order_polarity.flip_r threshold). Pearson r is
    offset-blind (see affine_stats) -- a component with a real instruction-driven mean shift and
    otherwise-identical shape reads as attr_r=1 here, i.e. "bypass"."""
    shared_comps, comps_norm, comps_inv, attr_norm, attr_inv, proj_norm, proj_inv, pair_uid = \
        _align_polarity(screen_norm, screen_inv, pairs_df)

    thresh = wbconfig.cfg()["order_polarity"]["flip_r"]
    rows = []
    for key, comp in shared_comps:
        jn, ji = comps_norm[(key, comp)], comps_inv[(key, comp)]
        a_n, a_i = attr_norm[:, jn], attr_inv[:, ji]
        p_n, p_i = proj_norm[:, jn], proj_inv[:, ji]
        attr_r, proj_r = _corr(a_n, a_i), _corr(p_n, p_i)
        rows.append({
            "key": key, "comp": comp,
            "attr_flip_rate": float(np.mean(np.sign(a_n) != np.sign(a_i))), "attr_r": attr_r,
            "proj_flip_rate": float(np.mean(np.sign(p_n) != np.sign(p_i))), "proj_r": proj_r,
            "n_rows": len(pair_uid), "class": _classify(attr_r, proj_r, thresh),
        })
    return pd.DataFrame(rows)


def polarity_affine(screen_norm: dict, screen_inv: dict, pairs_df: pd.DataFrame,
                     boot_n: int, seed: int) -> pd.DataFrame:
    """Per-component affine decomposition of attribution across polarity (affine_stats on
    per_prompt_attr, normal vs inverted) -- the primary polarity statistic feeding reclassify()."""
    shared_comps, comps_norm, comps_inv, attr_norm, attr_inv, _, _, pair_uid = \
        _align_polarity(screen_norm, screen_inv, pairs_df)
    boot_idx = _boot_pair_idx(pair_uid, boot_n, seed)

    rows = []
    for key, comp in shared_comps:
        jn, ji = comps_norm[(key, comp)], comps_inv[(key, comp)]
        stats = affine_stats(attr_norm[:, jn], attr_inv[:, ji], boot_idx)
        rows.append({"key": key, "comp": comp, "n_rows": len(pair_uid), **stats})
    return pd.DataFrame(rows)


def reclassify(order_reg: pd.DataFrame, affine: pd.DataFrame, screen_norm: dict,
               min_effect: float) -> pd.DataFrame:
    """Boolean-label reclassification from order_regression + polarity_affine -- the primary
    classification (the single-string legacy `class` from polarity_compare is merged in
    separately by main(), as class_legacy, for comparison). Gates:
    - instruction_sensitive: polarity mean_shift CI excludes 0 AND |effect| >= min_effect (an
      offset the old Pearson-r classifier is blind to -- see affine_stats).
    - slot: |beta_slot| > 2*SE(slot) AND |beta_slot| > |beta_winner|, AND attr_abs above the
      screen's median (a near-zero-attribution component shouldn't earn a position label just
      because its beta happens to clear a near-zero SE).
    - content: mirror of slot, on beta_winner.
    - unidentified: none of the above.
    Multi-label (boolean columns, not one string) -- a component can be both slot and
    instruction-sensitive. "The screen's median" = the median over the rows actually being
    classified here, i.e. order_reg INNER JOINED with affine (components present in both --
    polarity_affine can drop a few relative to order_reg's full top_k_components, see
    _align_polarity), not the full per-matrix SVD search space (screen_norm["attr_abs"] also
    carries every untouched, near-zero-activity component of every decomposed matrix -- a
    population-wide median there would sit near zero and the gate would rubber-stamp almost the
    whole top-k candidate set)."""
    abs_by_comp = {(key, i): float(v) for key, arr in screen_norm["attr_abs"].items()
                   for i, v in enumerate(arr)}

    df = order_reg.merge(affine, on=["key", "comp"])
    df["attr_abs"] = [abs_by_comp.get((k, c), float("nan")) for k, c in zip(df["key"], df["comp"])]
    above_median = df["attr_abs"] > df["attr_abs"].median()

    df["instruction_sensitive"] = (
        ((df["ci_lo"] > 0) | (df["ci_hi"] < 0)) & (df["effect"].abs() >= min_effect))
    slot_dom = (df["beta_slot"].abs() > 2 * df["se_slot"]) & (df["beta_slot"].abs() > df["beta_winner"].abs())
    winner_dom = (df["beta_winner"].abs() > 2 * df["se_winner"]) & (df["beta_winner"].abs() > df["beta_slot"].abs())
    df["slot"] = slot_dom & above_median
    df["content"] = winner_dom & above_median
    df["unidentified"] = ~(df["instruction_sensitive"] | df["slot"] | df["content"])
    return df


def migration_matched(classes_base: pd.DataFrame, classes_ft: pd.DataFrame, decompA, decompB) -> pd.DataFrame:
    """Base->FT component migration when base/ft do NOT share a decomp basis (different adapter,
    so index j in decompA's key is not the same direction as index j in decompB's key): Hungarian-
    matches components per key by before_after.match() instead of a naive (key, comp) index join,
    which would silently pair unrelated directions. Carries sim/low_sim through (before_after's
    own min_sim config) rather than dropping low-confidence matches -- inner-joined, so only
    components with a class label on both sides survive."""
    matched, _ = before_after.match(decompA, decompB)
    m = matched.merge(classes_base[["key", "comp", "class"]].rename(
        columns={"comp": "comp_a", "class": "class_base"}), on=["key", "comp_a"])
    m = m.merge(classes_ft[["key", "comp", "class"]].rename(
        columns={"comp": "comp_b", "class": "class_ft"}), on=["key", "comp_b"])
    m["changed"] = m["class_base"] != m["class_ft"]
    return m[["key", "comp_a", "comp_b", "sim", "low_sim", "class_base", "class_ft", "changed"]]


def class_distribution(classes_base: pd.DataFrame, classes_ft: pd.DataFrame) -> pd.DataFrame:
    """Tidy class-count/fraction distribution, base vs ft side by side (no cross-model component
    matching -- see main()'s migration-eligibility note)."""
    rows = []
    for side, df in (("base", classes_base), ("ft", classes_ft)):
        counts = df["class"].value_counts()
        total = int(counts.sum())
        for cls in CLASSES:
            n = int(counts.get(cls, 0))
            rows.append({"side": side, "class": cls, "n": n, "frac": n / total if total else 0.0})
    return pd.DataFrame(rows)


def fig_coherence(df, tag, n_outliers):
    """Scatter: coherence (x) vs order-anti-symmetric fraction |anti|/(|sym|+|anti|) (y) -- two
    independent order-anchoring proxies expected to move oppositely (slot-anchored components:
    high coherence + low anti fraction; transcript-anchored: low coherence + high anti fraction),
    so agreement shows up near y = 1 - x. Top |y-(1-x)| outliers labelled."""
    x = df["coherence"].to_numpy()
    denom = (df["abs_sym"] + df["abs_anti"]).replace(0, np.nan)
    y = (df["abs_anti"] / denom).to_numpy()

    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, color=wbfig.PALETTE[0], linewidths=0, label="component")
    ref = np.array([0.0, 1.0])
    ax.plot(ref, 1 - ref, color=wbfig.GRID, linewidth=1, zorder=0, label="y = 1 − x")
    dev = np.abs(y - (1 - x))
    dev_ranked = np.where(np.isnan(dev), -1.0, dev)
    for i in np.argsort(-dev_ranked)[:n_outliers]:
        ax.annotate(f"{df['key'].iloc[i]}#{df['comp'].iloc[i]}", (x[i], y[i]), fontsize=5)
    ax.set_xlabel("coherence = |attr_signed| / attr_abs")
    ax.set_ylabel("order-anti-symmetric fraction |anti| / (|sym| + |anti|)")
    ax.legend(frameon=False)
    wbfig.save(fig, f"order_polarity_{tag}_coherence")


def fig_polarity(classes, tag, thresh):
    """Scatter: attr_r (x) vs proj_r (y), quadrant lines at +-thresh, coloured by class."""
    fig = wbfig.new_fig(wbfig.COL, wbfig.COL)
    ax = fig.add_subplot(111)
    for cls in CLASSES:
        sub = classes[classes["class"] == cls]
        ax.scatter(sub["attr_r"], sub["proj_r"], s=8, color=CLASS_COLORS[cls], linewidths=0, label=cls)
    for v in (thresh, -thresh):
        ax.axvline(v, color=wbfig.GRID, linewidth=0.8, zorder=0)
        ax.axhline(v, color=wbfig.GRID, linewidth=0.8, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("attribution r (normal vs inverted)")
    ax.set_ylabel("projection r (normal vs inverted)")
    ax.legend(frameon=False, ncol=2)
    wbfig.save(fig, f"order_polarity_{tag}_polarity")


def fig_class_composition(classes, tag):
    """Stacked bars: class-fraction composition per layer, one panel per matrix kind present."""
    parsed = [importance_report._parse_key(k) for k in classes["key"]]
    df = classes.assign(layer=[p[0] for p in parsed], kind=[p[1] for p in parsed])
    kinds = [k for k in importance_report.KINDS if k in set(df["kind"])]

    fig = wbfig.new_fig(wbfig.WIDE, max(2.0, 1.8 * len(kinds)))
    for i, kind in enumerate(kinds):
        ax = fig.add_subplot(len(kinds), 1, i + 1)
        counts = df[df["kind"] == kind].groupby(["layer", "class"]).size().unstack("class")
        counts = counts.reindex(columns=CLASSES, fill_value=0).fillna(0.0).sort_index()
        frac = counts.div(counts.sum(axis=1), axis=0)
        bottom = np.zeros(len(frac))
        for cls in CLASSES:
            ax.bar(frac.index, frac[cls], bottom=bottom, color=CLASS_COLORS[cls], label=cls)
            bottom += frac[cls].to_numpy()
        ax.set_ylabel(kind)
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.legend(frameon=False, ncol=len(CLASSES), fontsize=6)
        if i == len(kinds) - 1:
            ax.set_xlabel("layer")
    wbfig.save(fig, f"order_polarity_{tag}_class_by_layer")


def fig_boolean_class_by_kind(reclass, tag):
    """Grouped bars: count of components with each boolean label True, by matrix kind (q/k/v/o/
    gate/up/down) -- primary class-distribution figure (fig_class_composition, above, is kept for
    the legacy single-string class)."""
    parsed = [importance_report._parse_key(k) for k in reclass["key"]]
    df = reclass.assign(kind=[p[1] for p in parsed])
    kinds = [k for k in importance_report.KINDS if k in set(df["kind"])]
    counts = df.groupby("kind")[list(BOOL_LABELS)].sum().reindex(kinds).fillna(0)

    fig = wbfig.new_fig(wbfig.WIDE, 3.0)
    ax = fig.add_subplot(111)
    x = np.arange(len(kinds))
    width = 0.8 / len(BOOL_LABELS)
    for i, lbl in enumerate(BOOL_LABELS):
        ax.bar(x + (i - (len(BOOL_LABELS) - 1) / 2) * width, counts[lbl], width=width,
               color=BOOL_COLORS[lbl], label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("n components")
    ax.legend(frameon=False, ncol=len(BOOL_LABELS))
    wbfig.save(fig, f"order_polarity_{tag}_class_by_kind")


def fig_bft_distribution(dist, tag):
    """Grouped bars: class-fraction distribution, base vs ft side by side."""
    pivot = dist.pivot(index="class", columns="side", values="frac").reindex(CLASSES)
    sides = [s for s in ("base", "ft") if s in pivot.columns]
    fig = wbfig.new_fig(wbfig.COL, 3.0)
    ax = fig.add_subplot(111)
    x = np.arange(len(pivot))
    width = 0.8 / len(sides)
    for i, side in enumerate(sides):
        ax.bar(x + (i - (len(sides) - 1) / 2) * width, pivot[side], width=width,
               color=wbfig.PALETTE[i], label=side)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("fraction of components")
    ax.legend(frameon=False)
    wbfig.save(fig, f"order_polarity_{tag}_bft_distribution")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-norm", required=True)
    parser.add_argument("--screen-inv", default=None)
    parser.add_argument("--screen-b-norm", default=None)
    parser.add_argument("--screen-b-inv", default=None)
    parser.add_argument("--decomp-a", default=None,
                         help="required for base<->ft migration when (model, adapter) differ")
    parser.add_argument("--decomp-b", default=None)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    assert (args.screen_b_norm is None) == (args.screen_b_inv is None), \
        "--screen-b-norm/--screen-b-inv must be given together"

    screen_norm = torch.load(args.screen_norm)
    pairs_df = wbpairs.load_pairs(c, split=screen_norm.get("split"), limit=screen_norm.get("limit"))

    out_dir = pathlib.Path(c["paths"]["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_df = order_split(screen_norm, pairs_df)
    order_reg = order_regression(screen_norm, pairs_df)
    main_df = split_df.merge(coherence(screen_norm), on=["key", "comp"]).merge(
        order_reg, on=["key", "comp"])
    main_df.to_csv(out_dir / f"order_polarity_{args.tag}.csv", index=False)
    fig_coherence(main_df, args.tag, c["order_polarity"]["n_outliers"])

    if args.screen_inv:
        screen_inv = torch.load(args.screen_inv)
        classes = polarity_compare(screen_norm, screen_inv, pairs_df)
        classes.to_csv(out_dir / f"order_polarity_{args.tag}_classes.csv", index=False)
        fig_polarity(classes, args.tag, c["order_polarity"]["flip_r"])
        fig_class_composition(classes, args.tag)

        affine = polarity_affine(screen_norm, screen_inv, pairs_df,
                                  c["order_polarity"]["boot_n"], c["order_polarity"]["boot_seed"])
        reclass = reclassify(order_reg, affine, screen_norm, c["order_polarity"]["min_effect"])
        reclass = reclass.merge(classes[["key", "comp", "class"]].rename(
            columns={"class": "class_legacy"}), on=["key", "comp"], how="left")
        reclass.to_csv(out_dir / f"order_polarity_{args.tag}_affine.csv", index=False)
        fig_boolean_class_by_kind(reclass, args.tag)

        if args.screen_b_norm:
            screen_b_norm, screen_b_inv = torch.load(args.screen_b_norm), torch.load(args.screen_b_inv)
            classes_b = polarity_compare(screen_b_norm, screen_b_inv, pairs_df)
            classes_b.to_csv(out_dir / f"order_polarity_{args.tag}_classes_b.csv", index=False)

            dist = class_distribution(classes, classes_b)
            dist.to_csv(out_dir / f"order_polarity_{args.tag}_class_dist.csv", index=False)
            fig_bft_distribution(dist, args.tag)

            # component-level migration needs a shared decomp basis: (key, comp) only lines up
            # directly when model AND adapter both match (same cache key in
            # screen._load_or_build_decomp) -- consistent with polarity_compare's own assert.
            # Otherwise decompA/decompB must be matched via before_after.match() first.
            same_basis = (screen_norm.get("model") == screen_b_norm.get("model")
                          and screen_norm.get("adapter") == screen_b_norm.get("adapter"))
            if same_basis:
                migration = classes.merge(classes_b, on=["key", "comp"], suffixes=("_base", "_ft"))
            else:
                assert args.decomp_a and args.decomp_b, (
                    "base/ft screens use different (model, adapter) -- (key, comp) indices are "
                    "NOT the same SVD basis; pass --decomp-a/--decomp-b so migration can match "
                    "components via before_after.match() instead of a naive index join.")
                migration = migration_matched(classes, classes_b, svd_components.load(args.decomp_a),
                                               svd_components.load(args.decomp_b))
            migration.to_csv(out_dir / f"order_polarity_{args.tag}_migration.csv", index=False)

    print(f"n_components={len(main_df)} -> {out_dir}")


if __name__ == "__main__":
    main()
