#!/usr/bin/env python3
"""Fit the judge GLM over the gpqa_tfa transcript features, one family at a time.

Why not one model with all 138 features: a joint fit over 138 correlated
predictors is poorly identified. Families are fitted separately instead, and the
one-at-a-time marginal fit is kept alongside so the attenuation from entering a
family jointly is visible rather than assumed.

Three specifications per family, which answer different questions:

    marginal   one feature at a time                  total association
    joint      the whole family together              association net of the family
    adjusted   joint, plus the capability gap         association net of the gap,
                                                      which is the bias estimate

The same feature contract is used in the separate capability analysis. Some
feature differences can therefore correlate with the capability gap. That is why
`adjusted` is reported next to `marginal` and not instead of it.

Usage:
    fit_features.py --csv analysis/hibayes/gpqa_correctness.csv --out-dir analysis/hibayes/features
    fit_features.py --bias-map analysis/hibayes/features   # after fitting
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcript_features  # noqa: E402 - sibling module, not an installed package

ROOT = Path(__file__).resolve().parents[2]

# The predictive checks and plots in config.yaml are dropped here: they cost a
# figure per fit and there are 152 fits. The convergence checks are kept, since
# an unnoticed divergence would corrupt the comparison across families.
CHECKS = ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi", "loo"]

CONDITION_ORDER = ["raw", "truncated_10", "truncated_50", "truncated_90"]

FIT = {"samples": 2000, "warmup": 2000, "chains": 4, "seed": 0, "target_accept": 0.95}

# Which slot the stronger model was shown in. Judges are at 0.506 when it is on
# the left and 0.707 when it is on the right, so side is the single largest
# effect in the corpus and larger than any feature. Left out, it inflates the
# residual and every feature interval with it.
POSITION = "major_is_right"

# Exact linear identities in the gpqa_tfa feature set. Each of these is one
# minus another feature, so a block containing both is rank deficient and the
# sampler wanders along a flat ridge: the process family fitted with both showed
# divergences and a failed BFMI. gpqa_tfa never saw this because it fits one
# feature at a time. Dropped from joint blocks only; marginal fits keep them,
# where a complement is perfectly interpretable on its own.
REDUNDANT = {
    "proc_conclusion_tail_frac": "1 - proc_first_conclusion_position",
    "repetition": "1 - ttr",
}


def build_config(csv: Path, predictors: list[str], outcome: str) -> dict:
    """A hibayes config for one predictor block, reusing the existing model."""
    intercept = "baseline_accuracy" if outcome == "is_correct" else "position_bias_left"
    return {
        "data_loader": {"paths": {"extracted_data": [str(csv)]}},
        "data_process": {
            "processors": [
                {"extract_observed_feature": {"feature_name": outcome}},
                {"extract_features": {
                    "categorical_features": ["judge", "item", "transcript_condition"],
                    "continuous_features": list(predictors),
                    "category_order": {"transcript_condition": list(CONDITION_ORDER)},
                }},
            ]
        },
        "model": {
            "path": "scripts/hibayes/model.py",
            "models": [{
                "name": "judge_feature_logistic_model",
                "config": {"predictors": list(predictors), "fit": dict(FIT)},
            }],
        },
        "check": {"checkers": list(CHECKS)},
        "communicate": {
            "path": "scripts/hibayes/communicators.py",
            "communicators": [{"judge_feature_report": {
                "intercept_label": intercept, "predictors": list(predictors),
                "make_plots": False,
            }}],
        },
        "platform": {"device_type": "cpu", "chain_method": "parallel"},
    }


def log(message: str) -> None:
    print(f"  note: {message}")


def specifications(columns: list[str], part: str, gap: str) -> dict[str, list[str]]:
    """Every predictor block to fit, keyed by a run name.

    `part` selects which component of each feature difference is entered:
    `within` and `between` fit one apiece, `both` fits the pair together, which
    is the unconstrained version of a single pooled coefficient.
    """
    wanted = ["within", "between"] if part == "both" else [part]
    grouped = {
        family: sorted(
            column for name in names for p in wanted
            if (column := f"d_{name}_{p}") in columns
        )
        for family, names in transcript_features.families().items()
    }
    runs: dict[str, list[str]] = {}
    for family, predictors in grouped.items():
        if not predictors:
            continue
        block = [p for p in predictors
                 if re.sub(r"^d_|_(within|between)$", "", p) not in REDUNDANT]
        for dropped in sorted(set(predictors) - set(block)):
            name = re.sub(r"^d_|_(within|between)$", "", dropped)
            log(f"{family}: dropping {name} from the joint block, it is "
                f"{REDUNDANT[name]}")
        runs[f"joint__{family}"] = block + [POSITION]
        runs[f"adjusted__{family}"] = block + [POSITION, gap]
        by_feature: dict[str, list[str]] = {}
        for predictor in predictors:
            by_feature.setdefault(re.sub(r"_(within|between)$", "", predictor), []).append(predictor)
        for feature, parts in by_feature.items():
            runs[f"marginal__{family}__{feature}"] = parts + [POSITION]
    return runs


KEY_COLUMNS = ["judge", "item", "transcript_condition", "model_pair",
               "self_comparison", "same_model_family", POSITION]


def complete_case_csv(csv: Path, out_dir: Path, predictors: list[str], gap: str,
                      outcome: str, subset: str | None = None) -> Path:
    """A copy of the data with only the columns a fit needs and no missing rows.

    Process features are NaN where a transcript is answer-only, and hibayes
    passes them straight into the design matrix, which makes every gradient NaN
    and the sampler fails outright rather than warning. Complete cases are the
    honest fix: the missingness is a property of the transcript, not of the
    outcome. Only the process family is affected on this corpus.
    """
    subset_column = subset.split("=", 1)[0] if subset else None
    needed = list(dict.fromkeys(
        c for c in [outcome, *KEY_COLUMNS, *predictors, gap, subset_column] if c
    ))
    columns = set(pd.read_csv(csv, nrows=0).columns)
    missing_columns = sorted(set(needed) - columns)
    if missing_columns:
        raise ValueError(f"{csv} is missing required columns: {missing_columns}")
    frame = pd.read_csv(csv, usecols=needed, low_memory=False)
    before = len(frame)
    frame = frame.dropna(subset=[c for c in predictors + [gap] if c in frame.columns])
    # Counted before the subset filter, or a restriction would be reported as
    # missingness.
    missing = before - len(frame)
    if subset:
        column, value = subset.split("=", 1)
        kept = frame[frame[column].astype(str) == value]
        log(f"subset {subset}: {len(kept):,} of {len(frame):,} rows, "
            f"{kept['model_pair'].nunique()} of {frame['model_pair'].nunique()} pairs")
        frame = kept
    if frame.empty:
        raise ValueError("no complete rows remain for this fit set")
    path = out_dir / "complete_cases.csv"
    frame.to_csv(path, index=False)
    if missing:
        log(f"dropped {missing:,} of {before:,} rows with a missing predictor "
            f"({missing / before:.2%})")
    return path


# A joint coefficient answers "this feature's effect net of the rest of the
# block". For a block that is one construct measured several ways the question
# has no unique answer, and the sampler splits the effect arbitrarily between the
# duplicates: length_structure produced log_n_words +0.92 against n_words -0.71
# with n_chars and n_words correlated at 0.992. Above this condition number the
# individual joint coefficients are not reported; the marginal fits are.
CONDITION_LIMIT = 15.0


def block_conditioning(data: pd.DataFrame, runs: dict[str, list[str]], gap: str) -> pd.DataFrame:
    """Condition number and worst correlated pair for every joint block."""
    rows = []
    for name, predictors in sorted(runs.items()):
        columns = [p for p in predictors if p not in (gap, POSITION) and p in data.columns]
        if not name.startswith("joint__") or len(columns) < 2:
            continue
        matrix = data[columns].dropna().to_numpy(float)
        spread = matrix.std(0)
        if not np.all(spread > 0):
            constant = [column for column, value in zip(columns, spread) if not value > 0]
            raise ValueError(
                f"{name} has constant predictors in this sample: {constant[:5]}"
            )
        matrix = (matrix - matrix.mean(0)) / spread
        singular = np.linalg.svd(matrix, compute_uv=False)
        correlation = np.corrcoef(matrix.T)
        np.fill_diagonal(correlation, 0.0)
        i, j = np.unravel_index(np.abs(correlation).argmax(), correlation.shape)
        rows.append({
            "block": name, "n_predictors": len(columns),
            "condition_number": float(singular[0] / singular[-1]),
            "worst_abs_r": float(abs(correlation[i, j])),
            "worst_pair": f"{columns[i]} ~ {columns[j]}",
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["joint_interpretable"] = frame["condition_number"] < CONDITION_LIMIT
    return frame


FACTOR_VARIANCE = 0.90


def factorize(data: pd.DataFrame, columns: list[str], prefix: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace a block with the orthogonal components that span it.

    Follows gpqa_tfa/src/stats/factor_check.py, which asks of its own correlated
    survivors whether they are one construct or many findings, and keeps however
    many components reach 90% of the variance. The components are orthogonal, so
    the block becomes perfectly conditioned and each coefficient has a single
    meaning again.

    The sign of a principal component is arbitrary, so each is oriented to make
    its largest loading positive. Without that the direction of an effect would
    change between runs on different data.
    """
    matrix = data[columns].to_numpy(float)
    spread = matrix.std(0)
    if not np.all(spread > 0):
        constant = [c for c, s in zip(columns, spread) if not s > 0]
        raise ValueError(
            f"cannot factorise {prefix}: {len(constant)} column(s) are constant in this "
            f"sample, e.g. {constant[:3]}. A between-part predictor is constant within a "
            "model pair, so this happens when a subset leaves too few pairs."
        )
    standardised = (matrix - matrix.mean(0)) / spread
    _, singular, right = np.linalg.svd(standardised, full_matrices=False)
    variance = singular ** 2 / (singular ** 2).sum()
    kept = int((variance.cumsum() < FACTOR_VARIANCE).sum() + 1)

    components = right[:kept].copy()
    for index, component in enumerate(components):
        if component[np.abs(component).argmax()] < 0:
            components[index] = -component

    scores = standardised @ components.T
    scores = scores / scores.std(0)
    names = [f"{prefix}_pc{i + 1}" for i in range(kept)]
    # Strip the prefix and whichever part suffix is present. A fixed -7 slice
    # only fits "_within"; "_between" is a character longer and left a trailing
    # underscore on every label in the between-part loadings table.
    labels = [re.sub(r"^d_|_(within|between)$", "", c) for c in columns]
    loadings = pd.DataFrame(components.T, index=labels, columns=names)
    loadings.loc["__variance_explained__"] = variance[:kept]
    return pd.DataFrame(scores, columns=names, index=data.index), loadings


def run_all(csv: Path, out_dir: Path, part: str, outcome: str, gap: str,
            subset: str | None = None) -> int:
    columns = list(pd.read_csv(csv, nrows=1).columns)
    if gap not in columns:
        raise SystemExit(f"{csv} has no column {gap}; run prepare_gpqa.py first")
    runs = specifications(columns, part, gap)
    if not runs:
        raise SystemExit(f"{csv} has no d_*_{part} columns; rerun prepare_gpqa.py with features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # One analysis sample for every fit, so coefficients are comparable across
    # families rather than each family silently using a different row set.
    every_predictor = sorted({p for block in runs.values() for p in block})
    data_csv = complete_case_csv(csv, out_dir, every_predictor, gap, outcome, subset)

    data = pd.read_csv(data_csv, low_memory=False)
    conditioning = block_conditioning(data, runs, gap)
    if not conditioning.empty:
        conditioning.to_csv(out_dir / "block_conditioning.csv", index=False)

    # A block that is one construct measured several ways is refitted on the
    # components that span it, so each joint coefficient means one thing. The
    # marginal fits keep the original features and are unaffected.
    added, loadings = {}, []
    ill_conditioned = (conditioning[~conditioning["joint_interpretable"]]
                       if not conditioning.empty else conditioning)
    for _, row in ill_conditioned.iterrows():
        family = row["block"].split("__")[1]
        columns = [p for p in runs[row["block"]] if p not in (gap, POSITION)]
        scores, block_loadings = factorize(data, columns, family)
        added.update({c: scores[c] for c in scores.columns})
        loadings.append(block_loadings)
        log(f"{family}: condition number {row['condition_number']:.1f} "
            f"({row['worst_pair']} at r={row['worst_abs_r']:.3f}); refitting the joint "
            f"block on {scores.shape[1]} component(s) spanning "
            f"{FACTOR_VARIANCE:.0%} of its variance")
        runs[f"joint__{family}"] = list(scores.columns) + [POSITION]
        runs[f"adjusted__{family}"] = list(scores.columns) + [POSITION, gap]
    if added:
        # Written beside the original rather than over it: a rerun must not
        # rewrite a file that fits already in flight are reading.
        data = pd.concat([data, pd.DataFrame(added, index=data.index)], axis=1)
        data_csv = out_dir / "complete_cases_factorised.csv"
        data.to_csv(data_csv, index=False)
        pd.concat(loadings, axis=1).round(4).to_csv(out_dir / "factor_loadings.csv")

    joint = sum(1 for k in runs if k.startswith("joint__"))
    print(f"{len(runs)} fits: {joint} joint, {joint} adjusted, {len(runs) - 2 * joint} marginal")
    for name, predictors in sorted(runs.items()):
        config_path = out_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(
            build_config(data_csv, predictors, outcome), sort_keys=False))
    print(f"wrote {len(runs)} configs to {out_dir}; nothing fitted")
    return 0


@functools.lru_cache(maxsize=None)
def _sha256(path: Path, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256(path, stat.st_size, stat.st_mtime_ns)


def result_matches_config(config: dict, run_dir: Path) -> bool:
    """Check complete provenance when available and legacy run logs otherwise."""
    snapshot = run_dir / "source_config.yaml"
    hashes = run_dir / "source_hashes.sha256"
    if snapshot.exists() or hashes.exists():
        if not snapshot.exists() or not hashes.exists():
            return False
        if yaml.safe_load(snapshot.read_text()) != config:
            return False
        seen = set()
        for line in hashes.read_text().splitlines():
            try:
                expected_hash, filename = line.split(maxsplit=1)
            except ValueError:
                return False
            path = (ROOT / filename.strip().lstrip("*")).resolve()
            if not path.is_file() or sha256(path) != expected_hash:
                return False
            seen.add(path)
        required = {
            (ROOT / path).resolve()
            for path in [
                config["model"]["path"], config["communicate"]["path"],
                *config["data_loader"]["paths"]["extracted_data"],
            ]
        }
        return required <= seen

    # Historical HiBayes outputs predate the source snapshot. Their process log
    # still records the input path, observed outcome, and categorical order.
    process_log = run_dir / "logs" / "logs_process.txt"
    if not process_log.exists():
        return False
    text = process_log.read_text(errors="replace")
    paths = config["data_loader"]["paths"]["extracted_data"]
    processors = config["data_process"]["processors"]
    outcome = processors[0]["extract_observed_feature"]["feature_name"]
    feature_config = processors[1]["extract_features"]
    if any(str(path) not in text for path in paths):
        return False
    if f"Extracted '{outcome}' -> 'obs'" not in text:
        return False
    return all(
        f"Category order: {list(order)!r}" in text
        for order in feature_config.get("category_order", {}).values()
    )


def collect(out_dir: Path) -> pd.DataFrame:
    """Gather the per-run population effects into one long table."""
    rows = []
    configs = sorted(out_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"no generated configs under {out_dir}")
    missing = []
    stale = []
    for config_path in configs:
        name = config_path.stem
        run_dir = out_dir / name
        table = run_dir / "communicate" / "population_effects.csv"
        model_config = run_dir / "models" / "judge_feature_logistic_model" / "model_config.json"
        if not table.exists() or not model_config.exists():
            missing.append(name)
            continue
        expected_config = yaml.safe_load(config_path.read_text())
        expected = expected_config["model"]["models"][0]["config"]
        actual = json.loads(model_config.read_text())
        actual_predictors = actual.get("extra_kwargs", {}).get("predictors")
        actual_fit = actual.get("fit", {})
        if (not result_matches_config(expected_config, run_dir) or
                actual_predictors != expected["predictors"] or
                any(actual_fit.get(key) != value for key, value in expected["fit"].items())):
            stale.append(name)
            continue
        spec, family, *rest = name.split("__")
        frame = pd.read_csv(table)
        frame["spec"], frame["family"] = spec, family
        rows.append(frame)
    if missing or stale:
        details = []
        if missing:
            details.append(f"missing {len(missing)} result(s), e.g. {missing[0]}")
        if stale:
            details.append(f"stale {len(stale)} result(s), e.g. {stale[0]}")
        raise SystemExit(f"cannot collect {out_dir}: {'; '.join(details)}")
    if not rows:
        raise SystemExit(f"no population_effects.csv under {out_dir}; nothing fitted yet")
    out = pd.concat(rows, ignore_index=True)
    # Keep the transcript features and the components that replaced a
    # rank-deficient block. Everything else the report emits is a control or an
    # intercept: the capability gap, the position indicator and the baseline have
    # no gpqa_tfa counterpart, and leaving them in labels a control as a bias.
    is_feature = out["effect"].str.startswith("d_")
    is_component = out["effect"].str.contains(r"_pc\d+$", regex=True)
    out = out[is_feature | is_component].copy()
    out["is_component"] = ~is_feature[is_feature | is_component].to_numpy()
    out["feature"] = (out["effect"].str.removeprefix("d_")
                      .str.replace(r"_(within|between|sd)$", "", regex=True))
    return out


def _normal_cdf(z: pd.Series) -> pd.Series:
    """Standard normal CDF without pulling scipy into this path."""
    from math import erf, sqrt
    return z.map(lambda v: np.nan if pd.isna(v) else 0.5 * (1 + erf(v / sqrt(2))))


def _benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH step-up q-values. A missing p is treated as 1, the conservative read."""
    filled = p.fillna(1.0)
    order = filled.sort_values()
    n = len(order)
    raw = order.to_numpy() * n / np.arange(1, n + 1)
    q = pd.Series(np.minimum.accumulate(raw[::-1])[::-1], index=order.index)
    return q.clip(upper=1.0).reindex(p.index)


def bias_map(out_dir: Path, method: str = "hand-built") -> pd.DataFrame:
    """Join the judge coefficients to gpqa_tfa's capability coefficients.

    The quadrant a feature falls in is the result: a feature the judge follows
    that does not track capability is a measured judge bias.
    """
    judge = collect(out_dir)
    capability = transcript_features.evidence_betas(method)
    merged = judge.merge(capability, on="feature", how="left")

    # Carry the conditioning verdict through, so a joint coefficient from a
    # rank-deficient block cannot be read as a finding without the caveat.
    conditioning_path = out_dir / "block_conditioning.csv"
    if conditioning_path.exists():
        conditioning = pd.read_csv(conditioning_path)
        conditioning["family"] = conditioning["block"].str.split("__").str[1]
        merged = merged.merge(
            conditioning[["family", "condition_number", "joint_interpretable"]],
            on="family", how="left",
        )
        merged["joint_interpretable"] = merged["joint_interpretable"].fillna(True)
        # Marginal fits have one predictor, so conditioning cannot apply to them.
        merged.loc[merged["spec"] == "marginal", "joint_interpretable"] = True
        # Components are the orthogonal replacement for an uninterpretable raw
        # block. Keep the source condition number, but mark the fitted component
        # coefficient itself as interpretable.
        merged.loc[merged["is_component"], "joint_interpretable"] = True
    # A credible interval excluding zero is not a multiplicity-controlled claim,
    # and at 91,481 rows the intervals are narrow enough that trivial effects
    # clear it. Report Benjamini-Hochberg both within a feature family and across
    # every feature of a specification. The global correction matches the scope
    # of the frozen capability table's hand-built-feature correction. The
    # posterior is summarised by a normal with the same mean and interval width,
    # which is what turns a Bayesian interval into a tail probability BH can use.
    spread = (merged["hdi_97.5%"] - merged["hdi_2.5%"]) / (2 * 1.959964)
    z = merged["mean_log_odds"].abs() / spread.replace(0, np.nan)
    merged["p_direction"] = 2 * (1 - _normal_cdf(z))
    merged["q_within_family"] = merged.groupby(["spec", "family"])["p_direction"].transform(
        _benjamini_hochberg)
    merged["q_global"] = merged.groupby("spec")["p_direction"].transform(_benjamini_hochberg)
    merged["judge_follows"] = merged["q_global"] < 0.05
    merged["credible_only"] = (merged["hdi_2.5%"] > 0) | (merged["hdi_97.5%"] < 0)
    merged["tracks_capability"] = merged["discovery"].fillna(False).astype(bool)
    merged["same_sign"] = (
        np.sign(merged["mean_log_odds"]) == np.sign(merged["beta_capability"])
    )

    def quadrant(follows: bool, tracks: bool, same_sign: bool, component: bool) -> str:
        if component:
            # A principal component is a combination of features, so gpqa_tfa has
            # no coefficient for it. Absence of a counterpart is not evidence that
            # it fails to track capability.
            return "component (no counterpart)"
        if not follows:
            return "missed signal" if tracks else "correctly ignored"
        if not tracks:
            return "JUDGE BIAS"
        # The judge reacts to a feature that does carry capability information,
        # but in the wrong direction. Worse than ignoring it.
        return "justified signal" if same_sign else "INVERTED"

    merged["quadrant"] = [
        quadrant(f, t, s, c) for f, t, s, c
        in zip(merged["judge_follows"], merged["tracks_capability"],
               merged["same_sign"], merged.get("is_component", False))
    ]
    unmatched = merged[~merged["is_component"] & merged["beta_capability"].isna()]
    if len(unmatched):
        log(f"{len(unmatched)} feature rows have no gpqa_tfa counterpart: "
            f"{sorted(unmatched['feature'].unique())[:5]}")
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("analysis/hibayes/gpqa_correctness.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/hibayes/features"))
    ap.add_argument("--part", default="within", choices=["within", "between", "both"],
                    help="which component of the feature difference to fit")
    ap.add_argument("--outcome", default="is_correct")
    ap.add_argument("--gap", default="capability_difference_major_sd")
    ap.add_argument("--subset", help="restrict rows, e.g. same_model_family=1")
    ap.add_argument("--bias-map", type=Path, help="collect finished fits into the bias map")
    ap.add_argument("--method", default="hand-built",
                    choices=["hand-built", "TF-IDF", "SAE"],
                    help="gpqa_tfa method whose capability betas the bias map joins")
    args = ap.parse_args(argv)

    if args.bias_map:
        table = bias_map(args.bias_map, args.method)
        path = args.bias_map / "bias_map.csv"
        table.to_csv(path, index=False)
        print(table["quadrant"].value_counts().to_string())
        print(f"\nwrote {path}")
        return 0

    return run_all(args.csv, args.out_dir, args.part, args.outcome, args.gap,
                   args.subset)


if __name__ == "__main__":
    raise SystemExit(main())
