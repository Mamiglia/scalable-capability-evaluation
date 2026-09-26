"""Rebuild the manuscript's results tables and figures: every analysis and figure script, in dependency order.

    uv run python scripts/analyses/reproduce_figures.py            # everything
    uv run python scripts/analyses/reproduce_figures.py --list     # the steps
    uv run python scripts/analyses/reproduce_figures.py --from signal_glm   # resume at a step

Reads judge logs (logs/), transcripts (SMALL_JUDGE_DATA) and embeddings (SMALL_JUDGE_EMBEDDINGS); launches no
judging. Scripts with a cached table (item/reference count, ensemble, cost, cross-domain) reuse it when present;
delete it, or run that script with --recompute, to rebuild it. The frontier figure runs only when FEATURES_ROOT
and OUT are set (see scripts/analyses/frontier_hidden_figure.py). Stops at the first failing step.
"""
import os
import subprocess
import sys

# (name, script and arguments); each step reads what the steps above it wrote
STEPS = [
    ("results", ["results.py"]),                              # analysis/results/forecasts.csv, metrics.csv
    ("likelihood_ratio", ["likelihood_ratio.py"]),            # likelihood_ratio.csv, cross_domain/transfer_likelihood_ratio.csv
    ("cross_domain", ["cross_domain.py"]),                    # domain transfer
    ("signal_glm", ["signal_glm.py"]),                        # domain GLM effects and contrasts, no self-judging
    ("generalisation", ["generalisation.py"]),                # capability generalisation
    ("channel_contributions", ["channel_contributions.py"]),  # pairwise channels
    ("position_thirds_removed", ["position_contributions.py"]),                   # pairwise thirds removed
    ("position_thirds_kept_alone", ["position_contributions.py", "--kept-alone"]),  # pairwise single thirds
    ("manuscript_error_figures", ["manuscript_error_figures.py"]),  # signal by protocol, time, prediction example, reasoning
    ("judge_not_reference", ["judge_not_reference.py"]),
    ("other_protocol_channels", ["other_protocol_channels.py"]),
    ("other_protocol_thirds", ["other_protocol_thirds.py"]),
    ("judge_choice", ["judge_choice.py"]),
    ("item_count_pairwise", ["item_count_effect.py"]),
    ("item_count_direct", ["item_count_effect.py", "--direct"]),
    ("item_count_both", ["item_count_effect.py", "--both"]),  # the paper figure, from the two tables above
    ("reference_count", ["reference_count_effect.py"]),
    ("judge_ensemble", ["judge_ensemble.py"]),
    ("protocol_cost", ["protocol_cost.py"]),
    ("judge_verdict_glm_build", ["judge_verdict_glm.py", "build"]),
    ("judge_verdict_glm_fit", ["judge_verdict_glm.py", "fit"]),
    ("judge_verdict_glm_summarise", ["judge_verdict_glm.py", "summarise"]),
    ("frontier_hidden", ["frontier_hidden_figure.py"]),
]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # every script reads repo-relative paths

if __name__ == "__main__":
    names = [name for name, _ in STEPS]
    if "--list" in sys.argv:
        print("\n".join(f"{name}: {' '.join(cmd)}" for name, cmd in STEPS))
        raise SystemExit(0)
    start = names.index(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH")]))}
    for name, (script, *args) in STEPS[start:]:
        if name == "frontier_hidden" and not ("FEATURES_ROOT" in env and "OUT" in env):
            print(f"== {name}: skipped (set FEATURES_ROOT and OUT)", flush=True)
            continue
        print(f"== {name}: {script} {' '.join(args)}", flush=True)
        if subprocess.run([sys.executable, os.path.join(HERE, script), *args], env=env, cwd=ROOT).returncode:
            raise SystemExit(f"{name} failed; resume with --from {name}")
