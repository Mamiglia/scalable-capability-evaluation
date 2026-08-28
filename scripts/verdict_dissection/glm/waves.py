"""Write one hibayes config per (model, feature wave, held-out design).

    uv run python scripts/verdict_dissection/glm/waves.py <grid_dir> [--models ...] [--waves ...] [--designs ...]

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
    "judge_verdict_from_pair": dict(target="pick_left", categorical=["judge", "bench"]),
}
FIT = {"samples": 500, "warmup": 500, "chains": 4, "seed": 0, "target_accept": 0.85}   # screening settings; ESS ~300-500


SURPRISAL_LOCAL = ["surprisal_local_mean__llama", "surprisal_rolling_short_position_adjusted_mean__llama"]
STRUCTURE = ["assistant_turns", "tool_calls", "tool_errors", "tool_outputs_empty", "submit_calls", "test_commands",
             "repeated_tool_calls", "distinct_tools", "prose_chars", "tool_call_chars", "tool_output_chars", "prose_share",
             "bare_answer", "display_equations", "inline_math", "option_references", "elimination_terms",
             "final_answer_restated", "bold_spans", "headers"]          # features_structure.py; present with predictors-structure.yaml
PAIRWISE = ["more_rigorous", "more_efficient", "fewer_errors", "deeper_understanding", "clearer", "better_verified",
            "more_complete", "better_tool_use"]        # annotate_pairs.py, signed per shown order
RUBRIC = ["reasoning_errors", "unjustified_claims", "redundant_steps", "self_corrections", "verifies_result",
          "explains_reasoning", "considers_alternatives", "wasted_actions", "clarity", "rigour", "efficiency", "depth"]   # annotate_transcripts.py


def waves(predictors):
    """Feature sets to fit, each an explicit list. Surprisal "local" needs only the 32-64 token window;
    "global" needs the full transcript and question context."""
    surface = [p for p in predictors if not p.startswith(("sae_", "qwen_dim_", "topo_")) and "surprisal" not in p
               and p != "n_words" and p not in STRUCTURE and p not in RUBRIC and p not in PAIRWISE]
    structure = [p for p in predictors if p in STRUCTURE]
    rubric = [p for p in predictors if p in RUBRIC]
    pairwise = [p for p in predictors if p in PAIRWISE]
    clause = [p for p in predictors if p.startswith("topo_")]           # clause-network graph shape; predictors-clause.yaml
    sae = [p for p in predictors if p.startswith("sae_shared_")]
    surprisal_local = [p for p in predictors if p in SURPRISAL_LOCAL]
    surprisal_global = [p for p in predictors if "surprisal" in p and p not in SURPRISAL_LOCAL]
    embedding = [p for p in predictors if p.startswith("qwen_dim_")]     # only present with prepare.py --dense
    return {**({"embedding": embedding} if embedding else {}),
        "length": ["log_chars"],
        "surface": surface,
        "sae": sae,
        "surface+sae": surface + sae,
        "surface+surprisal-local": surface + surprisal_local,
        "surface+surprisal": surface + surprisal_local + surprisal_global,
        "surface+sae+surprisal": surface + sae + surprisal_local + surprisal_global,
        **({"structure": structure, "surface+structure": surface + structure,
            "surface+sae+surprisal+structure": surface + sae + surprisal_local + surprisal_global + structure} if structure else {}),
        **({"clause": clause, "surface+clause": surface + clause,
            "surface+sae+surprisal+structure+clause": surface + sae + surprisal_local + surprisal_global + structure + clause}
           if clause else {}),
        **({"rubric": rubric, "surface+structure+rubric": surface + structure + rubric,
            "surface+sae+surprisal+structure+rubric": surface + sae + surprisal_local + surprisal_global + structure + rubric}
           if rubric else {}),
        **({"pairwise": pairwise, "rubric+pairwise": rubric + pairwise,
            "surface+structure+rubric+pairwise": surface + structure + rubric + pairwise,
            "surface+sae+surprisal+structure+rubric+pairwise": surface + sae + surprisal_local + surprisal_global + structure + rubric + pairwise}
           if pairwise else {}),
    }


BLOCK_ORDER = ["surface", "structure", "exposition", "derivation", "intermediate-errors", "clause-topology",
               "clause-descriptors", "clause-motifs", "clause-plan", "rubric", "sae", "surprisal", "pairwise"]


def block_waves(predictors, blocks=None):
    """Waves over named feature blocks (glm/blocks.yaml): each block alone, blocks added cumulatively in
    BLOCK_ORDER, everything, and everything minus one block at a time. Only blocks present in `predictors`."""
    blocks = blocks or yaml.safe_load((HERE / "blocks.yaml").read_text())
    present = {name: [p for p in members if p in predictors] for name, members in blocks.items()}
    present = {name: cols for name, cols in present.items() if cols}
    order = [b for b in BLOCK_ORDER if b in present] + [b for b in present if b not in BLOCK_ORDER]
    everything = [p for b in order for p in present[b]]
    out = {b: present[b] for b in order}
    running = []
    for b in order:
        running = running + present[b]
        if len(running) > len(present[b]):
            out["+".join(order[:order.index(b) + 1]) if order.index(b) < 3 else f"upto-{b}"] = list(running)
    out["all"] = everything
    for b in order:
        out[f"all-minus-{b}"] = [p for p in everything if p not in present[b]]
    return out


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
    parser.add_argument("--waves", nargs="+", help="feature sets to fit (default all)")
    parser.add_argument("--designs", nargs="+", default=list(designs()), help="held-out designs (default all)")
    args = parser.parse_args()
    predictors = yaml.safe_load((HERE / "predictors.yaml").read_text())
    jobs = []
    for model in args.models:
        for wave, kept in waves(predictors).items():
            if args.waves and wave not in args.waves:
                continue
            specs = {wave: False, f"{wave}+accuracy-gap": True} if model == "judge_verdict_from_pair" else {wave: False}
            for wave_name, use_gap in specs.items():
                for design in args.designs:
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
