"""Write one hibayes config per (model, ablation) for the cluster grid.

    uv run python scripts/verdict_dissection/glm/ablations.py <grid_dir>

Ablations, on the grouped predictor set: the full model; each predictor dropped on its own;
each feature block dropped; each surprisal scorer kept alone (other scorers dropped). Configs
go to <grid_dir>/configs/<model>/<ablation>.yaml and are listed, one per line, in
<grid_dir>/jobs.txt for the Slurm array. Fits use fewer draws than the headline fits and skip
LOO; held-out accuracy on the confirmation items is the comparison.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODELS = {
    "capability_from_transcript": dict(target="logit_model_accuracy", categorical=["bench", "item"], extra=[]),
    "stronger_model_from_pair": dict(target="left_stronger", categorical=["bench"], extra=[]),
    "judge_verdict_from_pair": dict(target="pick_left", categorical=["judge"], extra=["accuracy_difference"]),
}
SCORERS = ["llama", "gemma", "qwen"]
FIT = {"samples": 500, "warmup": 500, "chains": 4, "seed": 0, "target_accept": 0.9}


def ablation_sets(predictors, block_of):
    """Name -> predictor list for every ablation."""
    sets = {"full": predictors}
    for name in predictors:
        sets[f"drop_{name}"] = [p for p in predictors if p != name]
    for block in sorted({block_of[p] for p in predictors}):
        sets[f"drop_block_{block}"] = [p for p in predictors if block_of[p] != block]
    for scorer in SCORERS:
        others = [f"surprisal-{s}" for s in SCORERS if s != scorer]
        sets[f"only_scorer_{scorer}"] = [p for p in predictors if block_of[p] not in others]
    return sets


def config(model, predictors, input_path):
    spec = MODELS[model]
    return {
        "data_loader": {"paths": {"extracted_data": [str(input_path)]}},
        "data_process": {"processors": [
            {"extract_observed_feature": {"feature_name": spec["target"], "test": "test"}},
            {"extract_features": {"categorical_features": spec["categorical"],
                                  "continuous_features": spec["extra"] + predictors, "test": "test"}},
        ]},
        "model": {"path": "scripts/verdict_dissection/glm/model.py",
                  "models": [{"name": model, "config": {"predictors": predictors, "fit": FIT}}]},
        "check": {"checkers": ["divergences", "r_hat", "ess_bulk"]},
        "communicate": {"communicators": ["summary_table"]},     # the default plots are slow with 150 predictors
        "platform": {"device_type": "cpu", "chain_method": "parallel"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--inputs", type=Path, default=Path("judge_results/verdict_dissection/glm_inputs_grouped"))
    args = parser.parse_args()
    predictors = yaml.safe_load((HERE / "predictors-grouped.yaml").read_text())
    design_matrices = Path(yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())["output_root"])
    dictionary = pd.read_parquet(design_matrices / "grouped/feature-dictionary.parquet")
    block_of = dict(zip(dictionary.feature, dictionary.block))

    jobs = []
    for model in MODELS:
        for ablation, kept in ablation_sets(predictors, block_of).items():
            path = args.grid_dir / "configs" / model / f"{ablation}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config(model, kept, args.inputs / f"{model}.parquet"),
                                           sort_keys=False, width=100))
            jobs.append(f"{model}/{ablation}")
    (args.grid_dir / "jobs.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} configs -> {args.grid_dir}")


if __name__ == "__main__":
    main()
