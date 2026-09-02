"""Judge config for the steered-answer round.

Lists one comparison per steering condition against the unsteered
answers, for each judge, so the standard runner does the work. Routing
pins are copied from the pilot config so these verdicts are comparable
with the ones the directions were fit on: same provider, same
quantization, no fallbacks.

require_both_correct stays on, so every judged pair has both answers
correct and no verdict can ride on one being right and the other wrong.

Usage:
  uv run python scripts/analyses/judge_interp/make_judge_config.py \
      --logs data/steered_answers --out configs/judge_interp_steered.yaml
"""

import argparse
import glob
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PILOT_CONFIG = ROOT / "configs/gpqa_core_rd2_260803.yaml"

JUDGES = [
    "openrouter/qwen/qwen3-32b",          # the recorded judge, steered during fitting
    "openrouter/google/gemma-4-31b-it",   # second open-weights judge, never steered
    "openrouter/openai/gpt-4o-2024-08-06",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-root", default="logs/judge-interp-steered")
    ap.add_argument("--phase1-items", type=int, default=20)
    args = ap.parse_args()

    pilot = yaml.safe_load(PILOT_CONFIG.read_text())
    pins = {j: pilot["judge_model_args"][j] for j in JUDGES
            if j in pilot["judge_model_args"]}
    missing = [j for j in JUDGES if j not in pins]
    if missing:
        raise SystemExit(f"no routing pin in the pilot config for: {missing}")

    transcripts, conditions = {}, []
    for d in sorted(Path(args.logs).iterdir()):
        if not d.is_dir():
            continue
        evals = sorted(glob.glob(str(d / "*.eval")))
        if not evals:
            continue
        path = Path(evals[-1]).resolve()
        transcripts[d.name] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT)
                                  else path)
        if d.name != "unsteered":
            conditions.append(d.name)
    if "unsteered" not in transcripts:
        raise SystemExit("no unsteered log to compare against")

    # The runner passes this straight to inspect as the model, and shortens it
    # itself for log paths, so it has to be the full api/model path.
    comparisons = [[judge, cond, "unsteered"]
                   for cond in conditions for judge in JUDGES]
    cfg = dict(log_root=args.log_root, phase1_items=args.phase1_items,
               transcripts=transcripts,
               judge_model_args={j: pins[j] for j in JUDGES},
               comparisons=comparisons)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{len(conditions)} conditions x {len(JUDGES)} judges = "
          f"{len(comparisons)} comparisons")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
