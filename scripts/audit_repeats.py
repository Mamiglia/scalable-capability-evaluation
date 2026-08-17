"""Find repeat directories whose transcripts are copies of another repeat.

Until 2026-08-17 collect passed a month-long generate cache to inspect, so a repeat started
after (or racing) another one replayed its outputs on benchmarks with fixed prompts (aime,
mbpp; gpqa reshuffles options so it escaped). This lists, per <bench>/<hash>/<model>, the
share of identical completions between every pair of repeats.

    uv run python scripts/audit_repeats.py [data/aime2024 ...] [--min 0.5]
"""

import argparse
import itertools
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples


def completions(path: Path) -> dict:
    return {
        (s.id, s.epoch): s.output.completion
        for s in read_eval_log_samples(str(path), all_samples_required=False)
    }


def successful(repeat_dir: Path) -> Path | None:
    for path in repeat_dir.glob("*.eval"):
        try:
            if read_eval_log(str(path), header_only=True).status == "success":
                return path
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("data")])
    parser.add_argument("--min", type=float, default=0.0, help="Only report pairs at or above this identical share.")
    args = parser.parse_args()

    for root in args.roots:
        for model_dir in sorted(p for p in root.glob("**/") if p.parent.name and (p / "0").is_dir()):
            if "legacy" in model_dir.parts:
                continue
            logs = {int(d.name): successful(d) for d in model_dir.iterdir() if d.name.isdigit()}
            logs = {r: p for r, p in logs.items() if p}
            if len(logs) < 2:
                continue
            texts = {r: completions(p) for r, p in logs.items()}
            for a, b in itertools.combinations(sorted(texts), 2):
                common = texts[a].keys() & texts[b].keys()
                if not common:
                    continue
                share = sum(texts[a][k] == texts[b][k] for k in common) / len(common)
                if share >= args.min:
                    print(f"{share:5.0%}  {model_dir}  repeats {a} vs {b}  ({len(common)} samples)")


if __name__ == "__main__":
    main()
