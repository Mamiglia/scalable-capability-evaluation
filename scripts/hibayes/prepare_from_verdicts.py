#!/usr/bin/env python3
"""Build the judge GLM frame from exported pairwise verdicts.

Replaces prepare_gpqa.py, whose eval-log roots are gone. Row semantics come from
the verdict exporter: `order` is the position (1 or 2) holding the major model,
`picked` is the position the judge chose, so position 1 is the first-named
transcript ("left") in both. Lengths are the characters the judge saw.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from judge.dataset import is_effort_variant
from prepare_gpqa import model_family, standardise

LEFT_POSITION = 1
STANDARDISED = ("length_difference", "capability_difference")
# Columns of the style-rates file that are not lexicon categories.
STYLE_KEYS = ("model", "item", "n_chars", "reasoning_chars")
REASONING_MODES = ("none", "per_transcript", "per_model")
# Defaults are resolved against the repo root: a stale pre-length-fix copy of
# verdicts.csv sits under another script's directory, and a CWD-relative default
# would silently read it.
ROOT = Path(__file__).resolve().parents[2]
REASONING_CONFIG = ROOT / "configs" / "reasoning_filter.yaml"
VERDICTS = ROOT / "judge_results" / "judging_variations" / "verdicts.csv"
CAPABILITY_CONFIG = ROOT / "scripts" / "analyses" / "judge-elo" / "configs" / "gpqa.yaml"
CONDITION_CONFIGS = ROOT / "analyses" / "judge-variation" / "configs"
STYLE_RATES = ROOT / "analysis" / "hibayes" / "style_rates.csv"


def load_capabilities(config_path: Path, field: str) -> dict[str, float]:
    """model -> capability score from the judge-elo bench config."""
    config = yaml.safe_load(config_path.read_text())
    if field not in config:
        raise ValueError(f"{config_path} has no '{field}' block")
    return {model: float(value) for model, value in config[field].items()}


def check_conditions(bench: str, conditions: Sequence[str], config_dir: Path) -> None:
    """Reject conditions whose rows do not mean the same thing as the baseline's.

    Read from each condition's own collection config rather than a name list, so a
    new condition with the same property is caught too.
    """
    for condition in conditions:
        # the exporter labels native-reasoning runs `<condition>_native`, sharing the config
        path = config_dir / f"{bench}_{condition.removesuffix('_native')}.yaml"
        if not path.exists():
            raise ValueError(f"No config at {path}; cannot verify what condition {condition} means")
        config = yaml.safe_load(path.read_text())
        if config.get("invert_at_analysis"):
            raise ValueError(
                f"Condition {condition} asks the judge for the weaker model, so its capability "
                "and length slopes point the other way. The model has a per-condition intercept "
                "but no per-condition slope, so pooling these rows corrupts the shared slopes."
            )
        if config.get("include_reasoning"):
            raise ValueError(
                f"Condition {condition} shows the judge reasoning traces; this frame is "
                "final-text-only."
            )


def add_style_differences(frame: pd.DataFrame, style_rates: pd.DataFrame) -> list[str]:
    """Attach left-minus-right style-rate columns; return the difference names.

    n_chars is checked against the lengths the judge saw, so a rates file built
    from a different epoch or a text-only/reasoning mix cannot pass silently.
    """
    categories = [column for column in style_rates.columns if column not in STYLE_KEYS]
    if not categories:
        raise ValueError(f"Style rates have no category columns, only {list(style_rates.columns)}")
    rates = style_rates.set_index(["model", "item"])

    for side in ("left", "right"):
        keys = pd.MultiIndex.from_arrays([frame[f"{side}_model"], frame["item"]])
        missing = keys[~keys.isin(rates.index)].unique()
        if len(missing):
            raise ValueError(f"No style rates for {len(missing)} {side} transcripts, e.g. {missing[:3].tolist()}")
        matched = rates.reindex(keys)
        mismatched = (matched["n_chars"].to_numpy() != frame[f"{side}_chars"].to_numpy()).sum()
        if mismatched:
            raise ValueError(
                f"{mismatched} {side} transcripts differ in length from the ones the judge saw; "
                "the style rates were measured on different text"
            )
        for category in categories:
            frame[f"{side}_{category}"] = matched[category].to_numpy()

    differences = []
    for category in categories:
        name = f"{category}_difference"
        frame[name] = frame[f"left_{category}"] - frame[f"right_{category}"]
        differences.append(name)
    return differences


def keep_reasoning(
    frame: pd.DataFrame,
    style_rates: pd.DataFrame,
    mode: str,
    min_chars: int,
    effort_suffixes: Sequence[str],
) -> tuple[pd.Series, list[str], dict[str, int]]:
    """Row mask keeping comparisons whose two transcripts both show real work.

    per_transcript drops single transcripts under the floor. per_model drops an
    answer model whose median transcript is under it, keeping its models whole:
    no within-model selection, so the surviving transcripts of a weak model are
    still its typical ones. Effort variants go under either mode, so the pool
    holds one entry per model rather than one per effort setting.
    """
    if mode not in REASONING_MODES:
        raise ValueError(f"Unknown reasoning filter {mode}, have {list(REASONING_MODES)}")
    models = sorted(set(frame["left_model"]) | set(frame["right_model"]))
    missing = sorted(set(models) - set(style_rates["model"]))
    if missing:
        raise ValueError(f"No transcript lengths for {missing}; cannot judge their reasoning")

    dropped = {model for model in models if is_effort_variant(model, effort_suffixes)}
    counts = {"dropped_effort_variants": len(dropped)}
    if mode == "per_model":
        median = style_rates.groupby("model")["reasoning_chars"].median()
        dropped |= {model for model in models if median[model] < min_chars}
    counts["dropped_models"] = len(dropped)
    keep = ~(frame["left_model"].isin(dropped) | frame["right_model"].isin(dropped))

    if mode == "per_transcript":
        chars = style_rates.set_index(["model", "item"])["reasoning_chars"]
        thin = np.zeros(len(frame), dtype=bool)
        for side in ("left", "right"):
            keys = pd.MultiIndex.from_arrays([frame[f"{side}_model"], frame["item"]])
            side_chars = chars.reindex(keys)
            if side_chars.isna().any():
                raise ValueError(f"No transcript length for {int(side_chars.isna().sum())} {side} transcripts")
            thin |= side_chars.to_numpy() < min_chars
        keep &= ~thin
        counts["dropped_rows_thin_transcript"] = int(thin.sum())
    counts["dropped_rows_reasoning_filter"] = int((~keep).sum())
    return keep, sorted(dropped), counts


def build(
    verdicts: pd.DataFrame,
    capabilities: dict[str, float],
    bench: str,
    conditions: Sequence[str],
    condition_configs: Path = CONDITION_CONFIGS,
    style_rates: pd.DataFrame | None = None,
    reasoning_mode: str = "none",
    min_reasoning_chars: int = 0,
    effort_suffixes: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Verdict rows -> one model row per parsed verdict, with the GLM columns."""
    check_conditions(bench, conditions, condition_configs)
    data = verdicts[(verdicts["bench"] == bench) & (verdicts["condition"].isin(conditions))].copy()
    counts = {"verdicts_selected": len(data)}
    if data.empty:
        raise ValueError(f"No verdicts for bench {bench} and conditions {list(conditions)}")

    picked = pd.to_numeric(data["picked"], errors="coerce")
    parsed = picked.isin([1, 2])
    measured = data["len_major"].notna() & data["len_minor"].notna()
    counts["dropped_unparsed_verdict"] = int((~parsed).sum())
    counts["dropped_missing_length"] = int((parsed & ~measured).sum())
    data, picked = data[parsed & measured].copy(), picked[parsed & measured]

    missing = sorted((set(data["major"]) | set(data["minor"])) - set(capabilities))
    if missing:
        raise ValueError(f"Missing capability estimate for {missing}")

    # Compared numerically: a re-export writing "1"/"2" as strings would make every
    # raw comparison False and silently put the minor model on the left in every row.
    order = pd.to_numeric(data["order"], errors="coerce")
    if not order.isin([1, 2]).all():
        bad = sorted(set(data["order"][~order.isin([1, 2])].astype(str)))[:5]
        raise ValueError(f"order must be 1 or 2, got {bad}")
    major_is_left = order == LEFT_POSITION
    left_model = data["major"].where(major_is_left, data["minor"])
    right_model = data["minor"].where(major_is_left, data["major"])
    length_left = data["len_major"].where(major_is_left, data["len_minor"])
    length_right = data["len_minor"].where(major_is_left, data["len_major"])
    left_capability = left_model.map(capabilities)
    right_capability = right_model.map(capabilities)

    frame = pd.DataFrame(
        {
            "chosen_left": (picked == LEFT_POSITION).astype(int),
            "judge": data["judge"],
            "item": data["item_id"],
            "transcript_condition": data["condition"],
            "bench": data["bench"],
            "model_pair": [
                "|".join(sorted(pair)) for pair in zip(data["major"], data["minor"], strict=True)
            ],
            "left_model": left_model,
            "right_model": right_model,
            "order": order.astype(int),
            "picked": picked.astype(int),
            "left_chars": length_left.astype(int),
            "right_chars": length_right.astype(int),
            "left_capability": left_capability,
            "right_capability": right_capability,
            "length_difference": np.log1p(length_left) - np.log1p(length_right),
            "capability_difference": left_capability - right_capability,
            "same_model_family": (
                (left_model != right_model)
                & (left_model.map(model_family) == right_model.map(model_family))
            ).astype(int),
            "self_comparison": (left_model == right_model).astype(int),
        }
    )
    # Before standardising: the scales must come from the rows the model is fit on.
    if reasoning_mode != "none" or effort_suffixes:
        if style_rates is None:
            raise ValueError("The reasoning filter needs style rates for transcript lengths")
        keep, dropped, filter_counts = keep_reasoning(
            frame, style_rates, reasoning_mode, min_reasoning_chars, effort_suffixes
        )
        counts |= filter_counts
        counts["dropped_model_names"] = dropped
        frame = frame[keep].reset_index(drop=True)
        if frame.empty:
            raise ValueError("The reasoning filter removed every comparison")

    columns = list(STANDARDISED)
    if style_rates is not None:
        columns += add_style_differences(frame, style_rates)
    standardise(frame, columns)
    # The shared standardise() guards a zero scale but not a NaN one, which is what a
    # single-row or all-missing frame gives; NaN predictors would fit silently.
    scaled = [f"{column}_sd" for column in columns]
    if not np.isfinite(frame[scaled].to_numpy()).all():
        raise ValueError(f"Standardised columns are not finite: {scaled}")
    counts["model_rows"] = len(frame)
    frame = frame.sort_values(["judge", "model_pair", "item", "transcript_condition", "order"])
    return frame.reset_index(drop=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--capabilities", type=Path, default=CAPABILITY_CONFIG)
    parser.add_argument("--capability-field", default="eci", help="Block in the capabilities config to read")
    parser.add_argument("--condition-configs", type=Path, default=CONDITION_CONFIGS)
    parser.add_argument("--bench", default="gpqa")
    parser.add_argument("--conditions", nargs="+", default=["a_baseline"])
    parser.add_argument("--style-rates", type=Path, default=STYLE_RATES,
                        help="Per-transcript style rates from style_rates.py")
    parser.add_argument("--reasoning-config", type=Path, default=REASONING_CONFIG,
                        help="Defaults for the transcript filter")
    parser.add_argument("--reasoning-filter", choices=REASONING_MODES,
                        help="Drop transcripts, or whole answer models, that show no work")
    parser.add_argument("--min-reasoning-chars", type=int,
                        help="Characters after the ANSWER line a transcript needs")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "hibayes" / "gpqa_glm.csv")
    args = parser.parse_args()
    filter_config = yaml.safe_load(args.reasoning_config.read_text())
    mode = args.reasoning_filter or filter_config["mode"]
    min_chars = args.min_reasoning_chars if args.min_reasoning_chars is not None \
        else filter_config["min_reasoning_chars"]
    suffixes = filter_config["effort_suffixes"] if filter_config["exclude_effort_variants"] else ()
    print(f"Reading {args.verdicts}")
    print(f"Reasoning filter: {mode}, floor {min_chars} chars, effort suffixes {list(suffixes) or 'none'}")

    frame, counts = build(
        pd.read_csv(args.verdicts),
        load_capabilities(args.capabilities, args.capability_field),
        args.bench,
        args.conditions,
        args.condition_configs,
        pd.read_csv(args.style_rates),
        mode,
        min_chars,
        suffixes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")
    names = counts.pop("dropped_model_names", [])
    for name, value in counts.items():
        print(f"  {name}: {value:,}")
    if names:
        print(f"  dropped answer models: {', '.join(names)}")
    print(
        f"  judges: {frame['judge'].nunique()}, items: {frame['item'].nunique()}, "
        f"pairs: {frame['model_pair'].nunique()}, conditions: {frame['transcript_condition'].nunique()}"
    )
    print(f"  chosen_left rate: {frame['chosen_left'].mean():.4f}")


if __name__ == "__main__":
    main()
