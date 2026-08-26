"""Write one hibayes config per (model, feature wave, held-out design).

    uv run python scripts/verdict_dissection/glm/waves.py <grid_dir> [--models ...]

Waves add feature classes cumulatively: surface, + shared SAE, + surprisal. The judge model also
gets a "+accuracy-gap" variant of each wave (the capability-added specification). Configs go to
<grid_dir>/configs/<model>/<wave>/<design>.yaml, listed in <grid_dir>/jobs.txt for cluster.sbatch.
"""
import argparse
from pathlib import Path

import yaml

from prepare import designs

HERE = Path(__file__).resolve().parent
MODELS = {
    "capability_from_transcript": dict(target="logit_model_accuracy", categorical=["bench", "item"]),
    "stronger_model_from_pair": dict(target="left_stronger", categorical=["bench"]),
    "judge_verdict_from_pair": dict(target="pick_left", categorical=["judge"]),
}
FIT = {"samples": 500, "warmup": 500, "chains": 4, "seed": 0, "target_accept": 0.85}   # screening settings; ESS ~300-500


SURPRISAL_LOCAL = ["surprisal_local_mean__llama", "surprisal_rolling_short_position_adjusted_mean__llama"]


def waves(predictors):
    """Feature sets to fit, each an explicit list. Surprisal "local" needs only the 32-64 token window;
    "global" needs the full transcript and question context."""
    surface = [p for p in predictors if not p.startswith("sae_") and "surprisal" not in p and p != "n_words"]
    sae = [p for p in predictors if p.startswith("sae_shared_")]
    surprisal_local = [p for p in predictors if p in SURPRISAL_LOCAL]
    surprisal_global = [p for p in predictors if "surprisal" in p and p not in SURPRISAL_LOCAL]
    return {
        "surface": surface,
        "sae": sae,
        "surface+sae": surface + sae,
        "surface+surprisal-local": surface + surprisal_local,
        "surface+surprisal": surface + surprisal_local + surprisal_global,
        "surface+sae+surprisal": surface + sae + surprisal_local + surprisal_global,
    }


def config(model, predictors, input_path, use_accuracy_gap=False):
    spec = MODELS[model]
    continuous = predictors + (["accuracy_difference"] if use_accuracy_gap else [])
    model_config = {"predictors": predictors, "fit": FIT}
    if model == "judge_verdict_from_pair":
        model_config["use_accuracy_gap"] = use_accuracy_gap
    return {
        "data_loader": {"paths": {"extracted_data": [str(input_path)]}},
        "data_process": {"processors": [
            {"extract_observed_feature": {"feature_name": spec["target"], "test": "test"}},
            {"extract_features": {"categorical_features": spec["categorical"],
                                  "continuous_features": continuous, "test": "test"}},
        ]},
        "model": {"path": "scripts/verdict_dissection/glm/model.py",
                  "models": [{"name": model, "config": model_config}]},
        "check": {"checkers": ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi"]},
        "communicate": {"communicators": ["summary_table"]},
        "platform": {"device_type": "cpu", "chain_method": "parallel"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--inputs", type=Path, default=Path("judge_results/verdict_dissection/glm_inputs"))
    args = parser.parse_args()
    predictors = yaml.safe_load((HERE / "predictors.yaml").read_text())
    jobs = []
    for model in args.models:
        for wave, kept in waves(predictors).items():
            specs = {wave: False, f"{wave}+accuracy-gap": True} if model == "judge_verdict_from_pair" else {wave: False}
            for wave_name, use_gap in specs.items():
                for design in designs():
                    path = args.grid_dir / "configs" / model / wave_name / f"{design}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(
                        config(model, kept, args.inputs / design / f"{model}.parquet", use_gap),
                        sort_keys=False, width=100))
                    jobs.append(f"{model}/{wave_name}/{design}")
    (args.grid_dir / "jobs.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} configs -> {args.grid_dir}")


if __name__ == "__main__":
    main()
