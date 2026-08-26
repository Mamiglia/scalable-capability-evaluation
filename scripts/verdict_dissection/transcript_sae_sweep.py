"""Size sweep for the shared transcript SAE: fit each (width, top_k) in config sae.sweep on the
item-centred embeddings, then score the activations as pair regressors.

Per fit: held-out reconstruction, how many features fire on >= 5% of rows, mean active features
per transcript, and the accuracy of a logistic regression on left - right activations predicting
pick_left, cross-validated with items grouped, per benchmark group; optionally the same for
left_stronger (which side's model has the higher development solve rate; one row per transcript
pair, ties dropped) read from the pair design matrix. The same regression on the 2,560 centred
embedding dimensions is the dense reference. Development rows only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from transcript_sae import (apply_sae, balance_by_benchmark, centre_by_item, explained_variance,
                            fit_sae, load_dense_artifacts)
from transcript_sae_tables import KEYS, build_pairs

DEVELOPMENT = ("representation_train", "representation_validation")
PAIR_KEYS = ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right", "log_left", "log_right"]
C_GRID = (0.01, 0.1, 1.0)


def grouped_accuracy(features: np.ndarray, target: np.ndarray, groups: np.ndarray,
                     folds: int = 5, inner_folds: int = 3, seed: int = 0) -> float:
    """Held-out accuracy of a logistic regression, folds grouped by item. Columns are scaled by
    the training fold's SD (not centred: reversed pairs must stay negated); C is chosen on
    inner grouped folds."""
    predictions = np.zeros(len(target), dtype=bool)
    for train, test in GroupKFold(folds, shuffle=True, random_state=seed).split(features, target, groups):
        scale = features[train].std(axis=0) + 1e-8
        x_train, x_test = features[train] / scale, features[test] / scale
        best, best_score = None, -1.0
        for c in C_GRID:
            score = 0.0
            for fit, check in GroupKFold(inner_folds, shuffle=True, random_state=seed).split(x_train, target[train], groups[train]):
                model = LogisticRegression(C=c, max_iter=500).fit(x_train[fit], target[train][fit])
                score += model.score(x_train[check], target[train][check])
            if score > best_score:
                best, best_score = c, score
        model = LogisticRegression(C=best, max_iter=500).fit(x_train, target[train])
        predictions[test] = model.predict(x_test)
    return float((predictions == target).mean())


def score_block(pairs: pd.DataFrame, columns: list[str], group_of: dict[str, str],
                target: str = "pick_left") -> dict[str, float]:
    """Accuracy per benchmark group (fit groups from the config) and pooled over their rows."""
    out, correct, total = {}, 0, 0
    for group in sorted(set(group_of.values())):
        rows = pairs[pairs["bench"].map(group_of) == group]
        if not len(rows):
            continue
        accuracy = grouped_accuracy(rows[columns].to_numpy(np.float32), rows[target].to_numpy(bool),
                                    rows["item"].to_numpy())
        out[group] = accuracy
        correct += accuracy * len(rows)
        total += len(rows)
    out["pooled"] = correct / total
    return out


def transcript_frame(audit: pd.DataFrame, rows: pd.DataFrame, values: np.ndarray, prefix: str) -> pd.DataFrame:
    keyed = rows[["transcript_key", "split_role"]].merge(audit[["transcript_key", *KEYS]], on="transcript_key",
                                                        how="left", validate="one_to_one")
    columns = [f"{prefix}{i}" for i in range(values.shape[1])]
    return pd.concat([keyed[KEYS + ["split_role"]].reset_index(drop=True),
                      pd.DataFrame(values, columns=columns)], axis=1)


def development_pairs(verdicts: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    pairs = build_pairs(verdicts, frame)
    return pairs[pairs["split_role"].isin(DEVELOPMENT)].reset_index(drop=True)


def stronger_pairs(pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per transcript pair with a stronger-side label (left_stronger = +1 / -1)."""
    unique = pairs.drop_duplicates(PAIR_KEYS).merge(labels, on=PAIR_KEYS, how="inner", validate="one_to_one")
    unique = unique[unique["left_stronger"].notna()].reset_index(drop=True)
    return unique.assign(left_stronger=unique["left_stronger"] > 0)


def scores(pairs: pd.DataFrame, columns: list[str], group_of: dict[str, str], labels: pd.DataFrame | None) -> dict:
    out = {f"acc_{k}": v for k, v in score_block(pairs, columns, group_of).items()}
    if labels is not None:
        out |= {f"stronger_{k}": v for k, v in
                score_block(stronger_pairs(pairs, labels), columns, group_of, "left_stronger").items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/workstream_b.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="results table (csv)")
    parser.add_argument("--skip-dense", action="store_true", help="skip the 2,560-dim reference (slow)")
    parser.add_argument("--stronger-pairs", type=Path,
                        help="pair design matrix (build_design_matrices.py) with a left_stronger column")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["sae"]
    group_of = {bench: group for group, benches in config["fit_groups"].items() for bench in benches}

    rows, inputs, _ = load_dense_artifacts(args.dense_dir)
    audit = pd.read_parquet(args.dense_dir / "audit_rows.parquet")
    verdicts = pd.read_parquet(args.verdicts)
    labels = (pd.read_parquet(args.stronger_pairs, columns=PAIR_KEYS + ["left_stronger"]).drop_duplicates(PAIR_KEYS)
              if args.stronger_pairs else None)
    raw = np.asarray(inputs, dtype=np.float32)
    inputs = centre_by_item(raw, rows["item_group_key"].to_numpy())
    benchmarks = rows["benchmark"].to_numpy()
    train_mask = balance_by_benchmark(rows["split_role"].eq("representation_train").to_numpy(), benchmarks, config["seed"])
    validation_mask = balance_by_benchmark(rows["split_role"].eq("representation_validation").to_numpy(), benchmarks, config["seed"])

    results = []
    if not args.skip_dense:
        pairs = development_pairs(verdicts, transcript_frame(audit, rows, inputs, "dim_"))
        columns = [c for c in pairs if c.startswith("d_dim_")]
        results.append({"width": "dense", "top_k": "", "validation_ev": 1.0, "features_over_5pct": inputs.shape[1],
                        "mean_active": inputs.shape[1]} | scores(pairs, columns, group_of, labels))
        print(json.dumps(results[-1]), flush=True)
    for entry in config["sweep"]:
        fit_config = config | entry
        fit_inputs = inputs if fit_config.get("centre_by_item") else raw   # entries may switch centring off
        sae, history = fit_sae(fit_inputs[train_mask], fit_inputs[validation_mask], fit_config)
        activations, reconstructions = apply_sae(sae, fit_inputs)
        flipped = (activations < 0).sum(axis=0) > (activations > 0).sum(axis=0)
        activations[:, flipped] *= -1
        pairs = development_pairs(verdicts, transcript_frame(audit, rows, activations, "sae_"))
        columns = [c for c in pairs if c.startswith("d_sae_")]
        results.append({
            "width": entry["width"], "top_k": entry["top_k"], "centre_by_item": bool(fit_config.get("centre_by_item")),
            "validation_ev": explained_variance(fit_inputs[validation_mask], reconstructions[validation_mask]),
            "features_over_5pct": int(((activations != 0).mean(axis=0) >= 0.05).sum()),
            "mean_active": float((activations != 0).sum(axis=1).mean()),
            "epochs_run": len(history["train_loss"]),
        } | scores(pairs, columns, group_of, labels))
        print(json.dumps(results[-1]), flush=True)
    table = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
