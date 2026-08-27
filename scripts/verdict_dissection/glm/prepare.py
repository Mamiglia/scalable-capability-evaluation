"""Write the hibayes input tables: one per model and held-out design.

    uv run python scripts/verdict_dissection/glm/prepare.py <output_dir>

Tables go to <output_dir>/<design>/<model>.parquet with the predictors from predictors.yaml,
the target, the grouping columns and a boolean `test` column. Designs come from splits.yaml: model folds
hold out whole models on all items; the strict "new models and items" folds hold out unseen
models on unseen items and train only on seen models × seen items. A pair is held out when
either transcript is. Pair inputs rename the left-minus-right `d_` columns to
the bare feature names so all three models share one predictor list. Targets use the model's
accuracy on the benchmark over all its canonical rollouts (AIME years pooled, as in the design
matrices); the regressors are transcript features only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from build_design_matrices import pool_aime  # noqa: E402
HERE = Path(__file__).resolve().parent
SPLITS = yaml.safe_load((HERE / "splits.yaml").read_text())
PAIR_IDENTITY = ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right",
                 "log_left", "log_right"]
ACCURACY_CLIP = 0.01          # logit needs accuracies strictly inside (0, 1)


def designs():
    """Design name -> (held-out models, held-out item fold, strict). Model designs hold out whole models
    on all items. Strict designs ("new models and items") pair model fold r with item fold r: test rows
    have an unseen model on an unseen item, training rows a seen model on a seen item, the rest dropped."""
    out = {name: (models, None, False) for name, models in SPLITS["model_designs"].items()}
    folds = [name for name in SPLITS["model_designs"] if name.startswith("model-fold-")]
    out |= {f"new-models-and-items-{r}": (SPLITS["model_designs"][fold], r, True) for r, fold in enumerate(folds)}
    return out


def item_fold(bench, item):
    """Fold of each row's item: every k-th item in sorted order within its benchmark."""
    order = pd.DataFrame({"bench": bench, "item": item}).drop_duplicates().sort_values(["bench", "item"])
    order["fold"] = order.groupby("bench").cumcount() % SPLITS["item_folds"]
    return pd.DataFrame({"bench": bench, "item": item}).merge(order, how="left").fold.to_numpy()


def held_out(design, bench, item, *models):
    """Per row: 'test', 'train' or 'drop' for one design."""
    models_out, fold, strict = designs()[design]
    model_out = np.zeros(len(bench), bool)
    for column in models:
        model_out |= column.isin(models_out).to_numpy()
    if not strict:
        return np.where(model_out, "test", "train")
    item_out = item_fold(bench, item) == fold
    return np.select([model_out & item_out, ~model_out & ~item_out], ["test", "train"], "drop")


def capability_from_transcript(transcript, predictors, design):
    """One row per transcript; target is the logit of the model's accuracy on the benchmark."""
    accuracy = transcript.accuracy.clip(ACCURACY_CLIP, 1 - ACCURACY_CLIP)
    return pd.DataFrame({
        "bench": transcript.bench, "item": transcript.item, "model": transcript.model,
        "logit_model_accuracy": np.log(accuracy / (1 - accuracy)),
        "role": held_out(design, transcript.bench, transcript.item, transcript.model),
    }).join(transcript[predictors])


def stronger_model_from_pair(pair, predictors, design):
    """One row per distinct transcript pair (judges dropped), ties in accuracy dropped."""
    distinct = pair.drop_duplicates(PAIR_IDENTITY)
    distinct = distinct[distinct.accuracy_left != distinct.accuracy_right]
    return pd.DataFrame({
        "bench": distinct.bench, "item": distinct.item,
        "model_left": distinct.model_left, "model_right": distinct.model_right,
        "left_stronger": (distinct.accuracy_left > distinct.accuracy_right).astype(int),
        "role": held_out(design, distinct.bench, distinct.item, distinct.model_left, distinct.model_right),
    }).join(_bare_names(distinct, predictors))


def judge_verdict_from_pair(pair, predictors, design):
    """One row per judge draw. accuracy_difference is kept for the capability-added specification only."""
    return pd.DataFrame({
        "bench": pair.bench, "item": pair.item, "judge": pair.judge,
        "model_left": pair.model_left, "model_right": pair.model_right,
        "pick_left": pair.pick_left, "accuracy_difference": pair.accuracy_left - pair.accuracy_right,
        "role": held_out(design, pair.bench, pair.item, pair.model_left, pair.model_right),
    }).join(_bare_names(pair, predictors))


def with_sides(pair, transcript, predictors):
    """Each transcript's own (scaled) features attached to its side of the pair, as <feature>_left / _right."""
    keys = ["bench", "item", "model", "epoch", "log"]
    for side in ("left", "right"):
        side_table = transcript[keys + predictors].rename(
            columns={**{k: f"{k}_{side}" for k in ("model", "epoch", "log")}, **{f: f"{f}_{side}" for f in predictors}})
        pair = pair.merge(side_table, on=["bench", "item", f"model_{side}", f"epoch_{side}", f"log_{side}"],
                          how="left", validate="many_to_one")
    return pair


def _bare_names(pair, predictors):
    return pair[[f"d_{name}" for name in predictors]].set_axis(predictors, axis=1)


def with_accuracy(transcript, pair, accuracy):
    """Attach each model's benchmark accuracy (over all its rollouts) to transcripts and both pair sides.
    The design matrices' own leave-item-out accuracy columns are dropped first."""
    transcript = transcript.merge(accuracy, on=["bench", "model"], how="left", validate="many_to_one")
    pair = pair.drop(columns=["accuracy_left", "accuracy_right", "accuracy_difference", "left_stronger"], errors="ignore")
    for side in ("left", "right"):
        pair = pair.merge(accuracy.rename(columns={"model": f"model_{side}", "accuracy": f"accuracy_{side}"}),
                          on=["bench", f"model_{side}"], how="left", validate="many_to_one")
    if transcript.accuracy.isna().any() or pair.accuracy_left.isna().any() or pair.accuracy_right.isna().any():
        raise ValueError("a model is missing its benchmark accuracy")
    return transcript, pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--models", nargs="+", default=["capability_from_transcript", "stronger_model_from_pair", "judge_verdict_from_pair"])
    parser.add_argument("--dense", action="store_true", help="add the 2,560 dense embedding columns (qwen_dim_*) as predictors")
    parser.add_argument("--sides", action="store_true", help="judge tables also carry each side's own features as <feature>_left / _right")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    design_matrices = Path(config["output_root"])
    predictors = yaml.safe_load((HERE / "predictors.yaml").read_text())

    transcript = pd.read_parquet(design_matrices / "transcript.parquet")
    if args.dense:
        dense = [c for c in transcript.columns if c.startswith("qwen_dim_")]
        predictors = predictors + dense
        (HERE / "predictors-dense.yaml").write_text("# predictors.yaml plus the dense embedding columns (written by prepare.py --dense)\n"
                                                    + "".join(f"- {name}\n" for name in predictors))
    missing = [name for name in predictors if name not in transcript]
    if missing:
        raise ValueError(f"predictors not in the transcript design matrix: {missing}")
    pair = pd.read_parquet(design_matrices / "pair.parquet")
    identity = pool_aime(pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet",
                                         columns=["bench", "model", "item", "correct", "is_canonical"]))
    accuracy = (identity[identity.is_canonical].groupby(["bench", "model"], as_index=False).correct.mean()
                .rename(columns={"correct": "accuracy"}))
    transcript, pair = with_accuracy(transcript, pair, accuracy)

    if args.sides:
        pair = with_sides(pair, transcript, predictors)
    for design in designs():
        design_dir = args.output_dir / design
        design_dir.mkdir(parents=True, exist_ok=True)
        builders = {b.__name__: b for b in (capability_from_transcript, stronger_model_from_pair, judge_verdict_from_pair)}
        for build in (builders[name] for name in args.models):
            table = build(transcript if build is capability_from_transcript else pair, predictors, design)
            if args.sides and build is judge_verdict_from_pair:
                side_columns = [f"{name}_{side}" for side in ("left", "right") for name in predictors]
                table = table.join(pair.loc[table.index, side_columns])
            table = table[table.role != "drop"].assign(test=lambda t: t.role == "test").drop(columns="role")
            table.to_parquet(design_dir / f"{build.__name__}.parquet", index=False)
            print(f"{design} {build.__name__}: train {int((~table.test).sum())}, test {int(table.test.sum())}")


if __name__ == "__main__":
    main()
