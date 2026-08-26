"""Write hibayes configs for the feature waves: surface only, then + shared SAE, then + surprisal.

    uv run python scripts/verdict_dissection/glm/waves.py <wave_dir>

Each wave is fitted for all three models at the headline settings, on the ungrouped inputs
from prepare.py (judge_results/verdict_dissection/glm_inputs). Configs go to
<wave_dir>/configs/<model>/<wave>.yaml, listed in <wave_dir>/jobs.txt for cluster.sbatch.
The clause-network wave is added once its features exist.
"""
import argparse
from pathlib import Path

import yaml

from ablations import MODELS, config

HERE = Path(__file__).resolve().parent
FIT = {"samples": 1000, "warmup": 1000, "chains": 4, "seed": 0, "target_accept": 0.9}


def waves(predictors):
    surface = [p for p in predictors if not p.startswith("sae_") and "surprisal" not in p and p != "n_words"]
    sae = [p for p in predictors if p.startswith("sae_shared_")]
    surprisal = [p for p in predictors if "surprisal" in p]
    return {"surface": surface, "surface+sae": surface + sae, "surface+sae+surprisal": surface + sae + surprisal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wave_dir", type=Path)
    parser.add_argument("--inputs", type=Path, default=Path("judge_results/verdict_dissection/glm_inputs"))
    args = parser.parse_args()
    predictors = yaml.safe_load((HERE / "predictors.yaml").read_text())
    jobs = []
    for model in MODELS:
        for wave, kept in waves(predictors).items():
            path = args.wave_dir / "configs" / model / f"{wave}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            fit_config = config(model, kept, args.inputs / f"{model}.parquet")
            fit_config["model"]["models"][0]["config"]["fit"] = FIT
            fit_config["check"]["checkers"] = ["divergences", "r_hat", "ess_bulk", "ess_tail", "bfmi"]
            path.write_text(yaml.safe_dump(fit_config, sort_keys=False, width=100))
            jobs.append(f"{model}/{wave}")
    (args.wave_dir / "jobs.txt").write_text("\n".join(jobs) + "\n")
    sizes = {wave: len(kept) for wave, kept in waves(predictors).items()}
    print(f"{len(jobs)} configs -> {args.wave_dir}; predictors per wave {sizes}")


if __name__ == "__main__":
    main()
