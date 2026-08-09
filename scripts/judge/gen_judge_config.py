"""Generate a judge run config (run_judging.py schema) from a comparisons CSV.

Maps display names to configs/models shortnames, resolves each model's
transcript under --transcripts-dir/<shortname>/, and copies each judge's
model_args (endpoint pins) from configs/models/<shortname>.yml.

Usage:
  uv run python scripts/judge/gen_judge_config.py \
      --comparisons configs/comparisons/comparisons.csv \
      --transcripts-dir data/gpqa/<task_args_hash> \
      --log-root logs/data-expansion/gpqa \
      --out configs/judge_expansion_gpqa.yaml
"""

import argparse
import csv
from pathlib import Path

import yaml

MODELS_DIR = Path("configs/models")

# CSV display name -> configs/models shortname (also the collect dir name).
SHORTNAME = {
    "Qwen3 8B": "qwen3-8b",
    "Qwen3 32B": "qwen3-32b",
    "Qwen3 VL 30B A3B Instruct": "qwen3-vl-30b-a3b-instruct",
    "GPT-4.1": "gpt-4.1",
    "o4-mini": "o4-mini",
    "GLM-4.6": "glm-4.6",
    "Claude Sonnet 4.5": "claude-sonnet-4.5",
    "Claude Opus 4.7": "claude-opus-4.7",
    "Ling-2.6-flash": "ling-2.6-flash",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "DeepSeek V4 Flash": "deepseek-v4-flash-0731",
}


def shortname(display: str) -> str:
    try:
        return SHORTNAME[display]
    except KeyError:
        raise SystemExit(f"unknown model name in CSV: {display!r}")


def judge_slug_and_pins(short: str) -> tuple[str, dict]:
    cfg = yaml.safe_load((MODELS_DIR / f"{short}.yml").read_text())
    slug = "openrouter/" + cfg["name"].removeprefix("openrouter/")
    if not cfg.get("model_args"):
        raise SystemExit(f"configs/models/{short}.yml has no model_args (endpoint pin)")
    return slug, cfg["model_args"]


def transcript_path(transcripts_dir: Path, short: str) -> str:
    matches = sorted((transcripts_dir / short).glob("**/*.eval"))
    if len(matches) != 1:
        raise SystemExit(
            f"{short}: expected exactly 1 eval log under {transcripts_dir / short}, "
            f"found {len(matches)}: {[str(m) for m in matches]}"
        )
    return str(matches[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--transcripts-dir", type=Path, required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.comparisons.open()))

    members = sorted({shortname(r[c]) for r in rows for c in ("model_a", "model_b")})
    judges = sorted({shortname(r["judge"]) for r in rows})
    pins = {j: judge_slug_and_pins(j) for j in judges}  # short -> (slug, model_args)

    cfg = {
        "log_root": args.log_root,
        "transcripts": {m: transcript_path(args.transcripts_dir, m) for m in members},
        "comparisons": [
            [pins[shortname(r["judge"])][0], shortname(r["model_a"]), shortname(r["model_b"])]
            for r in rows
        ],
        "judge_model_args": {slug: margs for slug, margs in pins.values()},
    }

    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    print(f"{args.out}: {len(rows)} comparisons, {len(cfg['transcripts'])} transcripts, {len(judges)} judges")
    return 0


if __name__ == "__main__":
    main()