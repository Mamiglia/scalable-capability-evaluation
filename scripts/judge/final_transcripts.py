"""Transcript map (model config key -> rollout log) for judging after the data corrections.

For every model in configs/experiments/pairs/models.txt: the finalised log in its
data/<bench>/<collection>/<model dir>/<epoch>/ dir if there is one (finalize_rollout_logs.py
marks it with `finalised_from`), else the pinned log from --pins, else the latest finished
full log in the dir. Partial (sample-id) logs are never chosen.

Usage:
  uv run python scripts/judge/final_transcripts.py data/gpqa/b0a0d9c9 --epoch 0 \
      --pins configs/judge_more_pairs/transcripts/gpqa-epoch0.yaml --out <yaml>
"""

import argparse
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log

MODELS = Path("configs/experiments/pairs/models.txt")


def model_key_and_dir(model_yaml: Path) -> tuple[str, str]:
    name = yaml.safe_load(model_yaml.read_text())["name"]
    return model_yaml.stem, name.split("/")[-1].lower()


def pick(epoch_dir: Path, pinned: Path | None) -> Path | None:
    finals, fulls = [], []
    for path in sorted(epoch_dir.glob("*.eval")):
        if path.name.startswith("._"):
            continue
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        if header.eval.config.sample_id or header.eval.config.limit:
            continue  # sample-id re-runs and --limit smoke runs are never canonical
        if header.status == "success" and (header.eval.metadata or {}).get("finalised_from"):
            finals.append((header.eval.created, path))
        elif header.status == "success":
            fulls.append((header.results.completed_samples if header.results else 0, header.eval.created, path))
    if finals:
        return max(finals)[1]
    if pinned and pinned.is_file():
        try:
            header = read_eval_log(str(pinned), header_only=True)
        except Exception:
            header = None
        if (
            header
            and header.status == "success"
            and not header.eval.config.sample_id
            and not header.eval.config.limit
        ):
            return pinned
    return max(fulls)[2] if fulls else None  # most samples, then latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--pins", type=Path, help="yaml of model key -> log path used by earlier judging")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pins = {k: Path(v) for k, v in yaml.safe_load(args.pins.read_text()).items()} if args.pins else {}
    out = {}
    for line in MODELS.read_text().split():
        key, dirname = model_key_and_dir(Path(line))
        epoch_dir = args.collection / dirname / args.epoch
        chosen = pick(epoch_dir, pins.get(key)) if epoch_dir.is_dir() else None
        if chosen is None:
            print(f"NO LOG {key} ({epoch_dir})")
            continue
        out[key] = str(chosen)
        kind = "final" if "finalised" in str(read_eval_log(str(chosen), header_only=True).eval.metadata or {}) else ("pinned" if chosen == pins.get(key) else "latest full")
        print(f"{key:28s} {kind:11s} {chosen.name[:40]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out, width=200))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
