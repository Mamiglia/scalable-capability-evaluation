"""Rebuild the paper's results: every analysis and figure script, in dependency order.

    uv run python scripts/analyses/run.py                          # everything
    uv run python scripts/analyses/run.py --list                   # the steps
    uv run python scripts/analyses/run.py --from signal_glm        # resume at a step
    uv run python scripts/analyses/run.py --only results paper_tables   # just these steps

Inputs are the generality-labs Hugging Face datasets, checked out next to the code: data/ (benchmarks: frozen
tables and transcripts), logs/ (judging), glm_features/ and transcript-embeddings/ (or SMALL_JUDGE_DATA and
SMALL_JUDGE_EMBEDDINGS). SMALL_JUDGE_SEMESTERS (e.g. "2025-I 2025-II 2026-I") limits the tasks the results
runner loads. Launches no judging. Outputs go to analysis/: the paper figures and their .txt captions to
analysis/results/figures/paper/, the tables to analysis/results/paper_tables/. Steps whose table is cached
(cross-domain forecasts, item/reference count, ensemble, cost) reuse it when present; delete analysis/ for a
clean run. Stops at the first failing step.
"""
import os
import subprocess
import sys

# (name, script and arguments, what it makes); each step reads what the steps above it wrote
STEPS = [
    ("results", ["results.py"], "forecasts.csv: every forecast of every protocol, judge and condition"),
    ("likelihood_ratio", ["likelihood_ratio.py"], "likelihood_ratio.csv: per-target log likelihood ratios"),
    ("paper_tables", ["paper_tables.py"], "Table 3 (error per protocol and domain), Table 8 (candidate judges)"),
    ("signal_glm", ["signal_glm.py"], "Table 9, Figures 16-17 (domain GLM)"),
    ("channel_contributions", ["channel_contributions.py"], "Figure 5 (channels, pairwise judge)"),
    ("position_contributions", ["position_contributions.py"], "Figure 4 (thirds removed, pairwise judge)"),
    ("thirds_kept_alone", ["position_contributions.py", "--kept-alone"], "Figure 22 (single thirds, pairwise judge)"),
    ("generalisation", ["generalisation.py"], "Figure 8 (capability generalisation)"),
    ("manuscript_error_figures", ["manuscript_error_figures.py"], "Figures 3, 6, 7, 10 (signal, time, reasoning targets, example)"),
    ("frontier", ["frontier_hidden_figure.py"], "Figure 9 (closed-source targets)"),
    ("cross_domain", ["cross_domain.py"], "Figure 18 numbers and the Appendix D.5 tests"),
    ("judge_not_reference", ["judge_not_reference.py"], "Figure 21 (judge left out of its references)"),
    ("judge_choice", ["judge_choice.py"], "Figure 15 and the Appendix D.3 medians"),
    ("item_count_pairwise", ["item_count_effect.py"], "item-count table, pairwise"),
    ("item_count_direct", ["item_count_effect.py", "--direct"], "item-count table, absolute scoring"),
    ("item_count", ["item_count_effect.py", "--both"], "Figure 19 (number of items)"),
    ("reference_count", ["reference_count_effect.py"], "Figure 20 (number of references)"),
    ("features_reasoning_only", ["../../src/challenges/channels/features.py"], "deterministic features of the reasoning alone"),
    ("features_thirds", ["../../src/challenges/channels/features.py", "--thirds"], "deterministic features with a third cut"),
    ("other_protocol_thirds", ["other_protocol_thirds.py"], "Figure 23 (thirds, other protocols)"),
    ("other_protocol_channels", ["other_protocol_channels.py"], "Figure 24 (channels, other protocols)"),
    ("saturation_boundary", ["topk_refit.py"], "Figure 25, Table 10 (saturation boundary)"),
    ("protocol_cost", ["protocol_cost.py"], "Figure 27 (cost)"),
    ("judge_ensemble", ["judge_ensemble.py"], "Figure 28 (judge ensemble)"),
]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # every script reads repo-relative paths

if __name__ == "__main__":
    names = [name for name, *_ in STEPS]
    if "--list" in sys.argv:
        print("\n".join(f"{name}: {' '.join(cmd)}  ({what})" for name, cmd, what in STEPS))
        raise SystemExit(0)
    steps = STEPS[names.index(sys.argv[sys.argv.index("--from") + 1]):] if "--from" in sys.argv else STEPS
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1:]
        assert set(only) <= set(names), set(only) - set(names)
        steps = [s for s in STEPS if s[0] in only]
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH")]))}
    for name, (script, *args), what in steps:
        print(f"== {name}: {script} {' '.join(args)}  ({what})", flush=True)
        if subprocess.run([sys.executable, os.path.normpath(os.path.join(HERE, script)), *args], env=env, cwd=ROOT).returncode:
            raise SystemExit(f"{name} failed; resume with --from {name}")
