"""Generate per-judge judging configs (run_judging.py schema) from the pair configs.

For every configs/experiments/pairs/*.yml (or those listed in --pairs) whose two models
both have a finished transcript log for the epoch, one [judge, major, minor] row per
judge. Pairs with no both-correct instance are left out. The judges come from the
template's `judges` list (or its comparisons), with per-judge `judge_model_args`
(endpoint pins), `judge_generate_args` and `judge_max_connections` copied over.
A --transcripts map (model -> log path) pins the log a model is judged from; other
models get the largest finished log in their data dir.

Writes <out-dir>/epoch-<n>/<judge>.yaml, one file per judge so judges can run side by
side, and prints per-model solve counts and per-pair both-correct counts.

Usage:
  uv run python scripts/judge/gen_pairs_judge_config.py \
      --data-dir data/swe_bench_mini/477dc96c --epoch 0 \
      --template configs/judge_pilot_intercode_epoch0.yaml \
      --log-root logs/pilot-swe-mini/epoch-0 --out-dir configs/judge_pilot_swe_mini
"""

import argparse
import re
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log, read_eval_log_samples

from judge.util import extract_score

PAIRS_DIR = Path("configs/experiments/pairs")


def canonical_log(directory: Path) -> Path | None:
    """The transcript log for a model-epoch dir: the largest readable .eval, preferring
    status success. A run interrupted after its last sample keeps status "started" with
    all its samples intact, so status alone cannot decide."""
    status = {}
    for path in directory.glob("*.eval"):
        try:
            status[path] = read_eval_log(str(path), header_only=True).status
        except Exception:
            continue
    done = [p for p, s in status.items() if s == "success"]
    candidates = done or list(status)
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_size)
    if not done:
        print(f"WARNING {best}: status {status[best]!r}, no finished log in {directory}")
    return best


def correct_ids(path: Path) -> set[str]:
    return {str(s.id) for s in read_eval_log_samples(str(path), all_samples_required=False) if extract_score(s)}


def judge_overrides(
    template: dict, judges: list[str], judge: str, pod_max_connections: int
) -> dict:
    """Generation and concurrency settings for one generated judge config."""
    out = {}
    per_judge_generate = template.get("judge_generate_args") or {}
    per_judge_connections = template.get("judge_max_connections") or {}
    if judge in per_judge_generate:
        out["generate_args"] = per_judge_generate[judge]
    elif len(judges) == 1 and template.get("generate_args"):
        out["generate_args"] = template["generate_args"]
    if judge in per_judge_connections:
        out["max_connections"] = per_judge_connections[judge]
    elif len(judges) == 1 and template.get("max_connections"):
        out["max_connections"] = template["max_connections"]
    elif judge.startswith("openai-api/"):
        out["max_connections"] = pod_max_connections
    return out


def pair_models(pair_yml: Path) -> dict[str, tuple[str, str]]:
    """role -> (config key, data dir name) for the pair's major and minor."""
    out = {}
    for role, rel in re.findall(r"(major|minor):\s*!include\s+(\S+)", pair_yml.read_text()):
        model_yml = (pair_yml.parent / rel).resolve()
        name = yaml.safe_load(model_yml.read_text())["name"]
        out[role] = (model_yml.stem, name.split("/")[-1].lower())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="<benchmark>/<task_args_hash> dir")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--template", type=Path, required=True, help="config to copy judges and pins from")
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="file of pair names to include (default: all)")
    parser.add_argument("--judges", help="comma-separated judge slugs (default: all in the template)")
    parser.add_argument("--transcripts", type=Path, help="yaml of model -> log path to judge from")
    parser.add_argument("--pod-max-connections", type=int, default=8,
                        help="fallback max_connections for openai-api judges")
    args = parser.parse_args()

    template = yaml.safe_load(args.template.read_text())
    judges = template.get("judges") or list(dict.fromkeys(row[0] for row in template["comparisons"]))
    if args.judges:
        requested = args.judges.split(",")
        unknown = set(requested) - set(judges)
        if unknown:
            parser.error(f"--judges not present in template: {sorted(unknown)}")
        judges = requested
    pins = template.get("judge_model_args") or {}
    pinned_logs = (yaml.safe_load(args.transcripts.read_text()) or {}) if args.transcripts else {}
    pair_ymls = sorted(PAIRS_DIR.glob("*.yml"))
    if args.pairs:
        wanted = set(args.pairs.read_text().split())
        pair_ymls = [p for p in pair_ymls if p.stem in wanted]
        missing = wanted - {p.stem for p in pair_ymls}
        if missing:
            raise SystemExit(f"pairs not in {PAIRS_DIR}: {sorted(missing)}")

    logs, correct = {}, {}
    for pair_yml in pair_ymls:
        for key, dirname in pair_models(pair_yml).values():
            if key in logs:
                continue
            if key in pinned_logs and Path(pinned_logs[key]).is_file():
                logs[key] = Path(pinned_logs[key])
            else:
                if key in pinned_logs:
                    print(f"PINNED LOG GONE {key}: {pinned_logs[key]}")
                logs[key] = canonical_log(args.data_dir / dirname / str(args.epoch))
                if logs[key] is not None and pinned_logs:
                    print(f"NOT PINNED {key}: using {logs[key]}")
            if logs[key] is not None:
                correct[key] = correct_ids(logs[key])
    for key in sorted(logs):
        status = f"{len(correct[key])} correct" if logs[key] else "NO LOG"
        print(f"{key:42s} {status}")

    pairs, used = [], set()
    for pair_yml in pair_ymls:
        models = pair_models(pair_yml)
        major, minor = models["major"][0], models["minor"][0]
        if logs.get(major) is None or logs.get(minor) is None:
            print(f"SKIP {pair_yml.stem}: missing transcripts")
            continue
        both = len(correct[major] & correct[minor])
        if both == 0:
            print(f"SKIP {pair_yml.stem}: no both-correct instance")
            continue
        pairs.append((major, minor, both))
        used.update((major, minor))
    print(f"{len(pairs)} pairs, {2 * sum(b for _, _, b in pairs)} judge samples per judge")
    for major, minor, both in pairs:
        print(f"  {major}-vs-{minor}: {both}")

    out_dir = args.out_dir / f"epoch-{args.epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for judge in judges:
        cfg = {
            "log_root": args.log_root,
            "transcripts": {key: str(logs[key]) for key in sorted(used)},
            "comparisons": [[judge, major, minor] for major, minor, _ in pairs],
        }
        if judge in pins:
            cfg["judge_model_args"] = {judge: pins[judge]}
        cfg.update(judge_overrides(template, judges, judge, args.pod_max_connections))
        path = out_dir / f"{judge.split('/')[-1]}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
