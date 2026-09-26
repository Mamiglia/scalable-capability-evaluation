"""
Reproduce the paper's figures and tables from the hf data.

    uv run python scripts/analysis/run.py                        # everything, about three hours
    uv run python scripts/analysis/run.py --list                 # the steps and what each makes
    uv run python scripts/analysis/run.py --from generalisation  # resume at a step
    uv run python scripts/analysis/run.py --only capability_signal   # one step
    uv run python scripts/analysis/run.py --config my.yaml       # another configuration

Every setting is in scripts/analysis/config.yaml. Outputs go to analysis/: forecasts/ (the two tables every step
reads), one folder per group with its figures (each with a .txt of every plotted number and its test) and tables,
and cache/ (expensive intermediates, reused when present; delete it for a clean run). Launches no judging.

Figure and table numbers, and the output in analysis/<step>/ behind each:
    capability_signal Figure 3 error_by_protocol, Figure 10 prediction_example, Figure 16 domain_glm_effects, Figure 17
                      domain_glm_contrasts, Figure 21 no_self_judging; Table 3 error_by_domain.csv, Table 6
                      error_by_protocol.csv, Table 9 domain_glm_ratios.csv
    transcript_parts  Figure 4 thirds_removed, Figure 5 channel_shares and channels_removed, Figure 22 thirds_kept_alone,
                      Figure 23 thirds_other_protocols, Figure 24 channels_other_protocols
    generalisation    Figure 6 error_over_time, Figure 7 reasoning_targets, Figure 8 capability_distance, Figure 9
                      closed_targets, the numbers of Figure 18 and the cross-domain tests cross_domain_*.csv; Table 7
                      error_over_time.csv
    sensitivity       Figure 15 judge_choice, Figure 19 item_count, Figure 20 reference_count, Figure 25
                      saturation_boundary, Figure 27 cost, Figure 28 ensemble; Table 8 candidate_judges.csv, Table 10
                      saturation_boundary_pairwise.csv
Not included:
Figures 11 to 14 (scripts/analyses/domain_figures.py), Figure 26 (branch prediction-horizon)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# (step, runner, what it makes); each step reads what the steps above it wrote
STEPS = [
    ("forecasts", "forecasts.py", "analysis/forecasts: forecasts.csv, likelihood_ratio.csv"),
    ("features", "features.py", "per-transcript features of the thirds and channel conditions (analysis/cache)"),
    ("capability_signal", "capability_signal.py", "analysis/capability_signal: is there capability information in saturated transcripts?"),
    ("transcript_parts", "transcript_parts.py", "analysis/transcript_parts: where in the transcript is it?"),
    ("generalisation", "generalisation.py", "analysis/generalisation: does it hold beyond the reference models?"),
    ("sensitivity", "sensitivity.py", "analysis/sensitivity: how much do the analysis choices matter?"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    parser.add_argument("--from", dest="start", choices=[s for s, *_ in STEPS], help="resume at this step")
    parser.add_argument("--only", nargs="+", choices=[s for s, *_ in STEPS], help="run only these steps")
    args = parser.parse_args()
    if args.list:
        print("\n".join(f"{step:17s} {runner:22s} {what}" for step, runner, what in STEPS))
        raise SystemExit(0)
    names = [s for s, *_ in STEPS]
    steps = [s for s in STEPS if (args.only is None or s[0] in args.only) and names.index(s[0]) >= names.index(args.start or names[0])]
    for step, runner, what in steps:
        print(f"== {step}: {what}", flush=True)
        # -P: the runner's own folder is not put first on the import path
        if subprocess.run([sys.executable, "-P", str(HERE / runner), "--config", args.config], cwd=ROOT).returncode:
            raise SystemExit(f"{step} failed; resume with --from {step}")
