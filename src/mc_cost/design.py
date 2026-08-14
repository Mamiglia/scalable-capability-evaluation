from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalogue import pool_enabled


@dataclass(frozen=True)
class AutomaticConfig:
    j: int
    m: int
    n: int
    attempted_items: int
    presentations: int = 2
    selected_judges: tuple[str, ...] = ()
    require_stronger: bool = True


def automatic_comparisons(catalogue: pd.DataFrame, config: AutomaticConfig, rng) -> pd.DataFrame:
    required = {"model_id", "release", "theta", "judge_pool", "candidate_pool"}
    missing = sorted(required - set(catalogue))
    if missing:
        raise ValueError(f"catalogue lacks required columns: {', '.join(missing)}")
    if catalogue["model_id"].duplicated().any():
        raise ValueError("duplicate model_id")
    for name, value in (("J", config.j), ("M", config.m), ("N", config.n),
                        ("attempted_items", config.attempted_items), ("presentations", config.presentations)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.require_stronger) is not bool:
        raise ValueError("require_stronger must be a bool")
    selected_judges = config.selected_judges
    if (not isinstance(selected_judges, Sequence)
            or isinstance(selected_judges, (str, bytes, bytearray))
            or any(not isinstance(model, str) or not model for model in selected_judges)):
        raise ValueError("selected_judges must be a non-string sequence of non-null strings")
    selected_judges = tuple(selected_judges)

    models = catalogue.copy()
    models["release_day"] = pd.to_datetime(models["release"], errors="coerce")
    models["theta_value"] = pd.to_numeric(models["theta"], errors="coerce")
    models["judge_enabled"] = models["judge_pool"].map(pool_enabled)
    models["candidate_enabled"] = models["candidate_pool"].map(pool_enabled)
    valid_judges = models.loc[
        models["judge_enabled"] & models["release_day"].notna() & np.isfinite(models["theta_value"])
    ]
    if selected_judges:
        selected = list(selected_judges)
        if len(selected) != config.j or len(set(selected)) != len(selected):
            raise ValueError(f"requested J={config.j}, selected_judges must contain that many distinct identities")
        valid_ids = set(valid_judges["model_id"])
        invalid = [model for model in selected if model not in valid_ids]
        if invalid:
            raise ValueError(f"selected judges are not eligible judge-pool models: {', '.join(map(str, invalid))}")
        judges = valid_judges.set_index("model_id").loc[selected].reset_index()
    else:
        if len(valid_judges) < config.j:
            raise ValueError(f"requested J={config.j}, eligible judge-pool count={len(valid_judges)}")
        selected = rng.choice(valid_judges["model_id"].to_numpy(), config.j, replace=False).tolist()
        judges = valid_judges.set_index("model_id").loc[selected].reset_index()

    anchors = models.loc[
        models["candidate_enabled"] & models["release_day"].notna() & np.isfinite(models["theta_value"])
    ]
    anchor_mask = anchors["release_day"].gt(judges["release_day"].max())
    if config.require_stronger:
        anchor_mask &= anchors["theta_value"].gt(judges["theta_value"].max())
    eligible_anchors = anchors.loc[anchor_mask]
    if len(eligible_anchors) < config.m:
        restriction = "newer and stronger than every selected judge" if config.require_stronger else "newer than every selected judge"
        raise ValueError(f"requested M={config.m}, eligible anchor count={len(eligible_anchors)} ({restriction})")
    anchors = rng.choice(eligible_anchors["model_id"].to_numpy(), config.m, replace=False).tolist()

    comparator_sets = {}
    release_by_id = models.set_index("model_id")["release_day"]
    for anchor in anchors:
        comparators = models.loc[models["release_day"].lt(release_by_id[anchor]), "model_id"]
        if len(comparators) < config.n:
            raise ValueError(
                f"requested N={config.n}, eligible comparator count={len(comparators)} "
                f"for anchor {anchor} (release before anchor)"
            )
        comparator_sets[anchor] = rng.choice(comparators.to_numpy(), config.n, replace=False).tolist()
    return pd.DataFrame(
        [{"judge": judge, "m1": anchor, "m2": comparator}
         for judge in selected for anchor in anchors for comparator in comparator_sets[anchor]],
        columns=["judge", "m1", "m2"],
    )


def manual_comparisons(catalogue: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"judge", "m1", "m2"}
    optional = {"judge_reasoning_class", "m1_reasoning_class", "m2_reasoning_class"}
    columns = list(rows.columns)
    if set(columns) - required - optional or required - set(columns) or len(set(columns)) != len(columns):
        raise ValueError("manual comparison columns must be judge,m1,m2 with optional *_reasoning_class overrides")
    if "model_id" not in catalogue or catalogue["model_id"].duplicated().any():
        raise ValueError("catalogue must contain unique model_id values")

    models = catalogue.copy().set_index("model_id", drop=False)
    models["release_day"] = pd.to_datetime(models.get("release", pd.Series(index=models.index, dtype=object)), errors="coerce")
    models["theta_value"] = pd.to_numeric(models.get("theta", pd.Series(index=models.index, dtype=object)), errors="coerce")
    records, warnings = [], []
    for output_row, (source_row, row) in enumerate(rows.iterrows()):
        identities = {role: row[role] for role in ("judge", "m1", "m2")}
        unknown = [str(model) for model in identities.values() if pd.isna(model) or model not in models.index]
        if unknown:
            raise ValueError(f"unknown model identity: {', '.join(unknown)}")
        classes = {}
        for role, model in identities.items():
            override = row.get(f"{role}_reasoning_class", pd.NA)
            if pd.isna(override):
                value = models.at[model, "reasoning_class"] if "reasoning_class" in models else pd.NA
            elif isinstance(override, str) and override.strip().lower() in {"reasoning", "non_reasoning"}:
                value = override.strip().lower()
            else:
                raise ValueError(f"{role}_reasoning_class must be reasoning or non_reasoning")
            classes[role] = value

        missing = []
        for role, model in identities.items():
            metadata = models.loc[model]
            if not np.isfinite(metadata["theta_value"]):
                missing.append(f"{role}.theta")
            if pd.isna(metadata["release_day"]):
                missing.append(f"{role}.release")
            if pd.isna(classes[role]):
                missing.append(f"{role}.reasoning_class")
        predictable = not missing
        records.append({**identities, **{f"{role}_reasoning_class": classes[role] for role in identities},
                        "predictable": predictable})
        if missing:
            warnings.append({"code": "missing_prediction_metadata", "row": output_row, "source_row": source_row,
                             "message": f"missing prediction metadata: {', '.join(missing)}"})

        judge, m1, m2 = (models.loc[identities[role]] for role in ("judge", "m1", "m2"))
        violations = []
        if pd.notna(judge.release_day) and pd.notna(m1.release_day) and m1.release_day <= judge.release_day:
            violations.append("m1 is not newer than judge")
        if np.isfinite(judge.theta_value) and np.isfinite(m1.theta_value) and m1.theta_value <= judge.theta_value:
            violations.append("m1 is not stronger than judge")
        if pd.notna(m1.release_day) and pd.notna(m2.release_day) and m2.release_day >= m1.release_day:
            violations.append("m2 is not older than m1")
        if violations:
            warnings.append({"code": "automatic_rule_violation", "row": output_row, "source_row": source_row,
                             "message": "; ".join(violations)})

        for role, pool in (("judge", "judge_pool"), ("m1", "candidate_pool")):
            value = models.at[identities[role], pool] if pool in models else pd.NA
            if pd.notna(value) and not pool_enabled(value):
                warnings.append({"code": "out_of_support", "row": output_row, "source_row": source_row,
                                 "message": f"{role} is outside the automatic {pool} support"})
    return (
        pd.DataFrame(records, columns=["judge", "m1", "m2", "judge_reasoning_class", "m1_reasoning_class",
                                       "m2_reasoning_class", "predictable"]),
        pd.DataFrame(warnings, columns=["code", "row", "source_row", "message"]),
    )
