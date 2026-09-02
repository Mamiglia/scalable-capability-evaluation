"""Package the intercode judging into the LASR-G5/judging pilot layout.

Local runs split each (judge, pair) across epoch-<n>/phase-<p> sub-eval-sets, and
stall-retries leave duplicate .eval files. The pilot format (pilot/<bench>/<judge>/
<major-vs-minor>/) has one .eval per rollout epoch per pair plus eval-set.json,
logs.json, .eval-set-id. This merges the two phases per epoch into a single log
(recompute_metrics rebuilds the results block over all samples), dedupes retries by
keeping the newest successful log per task_id, and writes the pilot layout to
--out for upload.

Usage:
  uv run python scripts/judge/package_pilot_upload.py --out tmp/pilot-intercode-upload
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from inspect_ai.log import read_eval_log, write_eval_log, recompute_metrics

import judge.scorer  # noqa: F401  registers the custom metrics recompute_metrics needs

SRC = Path("logs/pilot-intercode")
EPOCHS = (0, 1)
PHASES = ("phase-1", "phase-2")


def newest_success(paths: list[str]):
    """Newest successful log among retries of one task (filenames start with an ISO ts)."""
    ok = [p for p in paths if read_eval_log(p, header_only=True).status == "success"]
    return max(ok, key=lambda p: Path(p).name) if ok else None


def merge_epoch(judge: str, pair: str, epoch: int):
    """One EvalLog over all samples of a (judge, pair, epoch), phases combined."""
    logs = []
    for phase in PHASES:
        d = SRC / f"epoch-{epoch}" / judge / pair / phase
        hits = sorted(glob.glob(f"{d}/*.eval"))
        if not hits:
            continue
        by_task = defaultdict(list)
        for p in hits:
            by_task[read_eval_log(p, header_only=True).eval.task_id].append(p)
        for paths in by_task.values():
            chosen = newest_success(paths)
            if chosen:
                logs.append(read_eval_log(chosen))
    if not logs:
        return None
    base = logs[0]
    for other in logs[1:]:
        base.samples.extend(other.samples)
        base.reductions[0].samples.extend(other.reductions[0].samples)
    # Distinct sample ids only (guards against a phase boundary overlap).
    seen, samples = set(), []
    for s in base.samples:
        key = (s.id, s.epoch)
        if key not in seen:
            seen.add(key)
            samples.append(s)
    base.samples = samples
    base.results.total_samples = len(samples)
    base.results.completed_samples = len(samples)
    recompute_metrics(base)
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bench", default="intercode")
    args = ap.parse_args()

    judges = sorted(d.name for d in (SRC / "epoch-0").iterdir() if d.is_dir())
    total_logs = 0
    for judge in judges:
        pairs = sorted(d.name for d in (SRC / "epoch-0" / judge).iterdir() if d.is_dir())
        for pair in pairs:
            merged = {ep: merge_epoch(judge, pair, ep) for ep in EPOCHS}
            merged = {ep: log for ep, log in merged.items() if log is not None}
            if not merged:
                continue
            out_dir = args.out / "pilot" / args.bench / judge / pair
            out_dir.mkdir(parents=True, exist_ok=True)
            # Pilot task name: <bench>/<major>|<minor> (no judge prefix), +seq per epoch.
            major, minor = pair.split("-vs-")
            base_task = f"{args.bench}/{major}|{minor}"
            logs_manifest, tasks = {}, []
            for ep, log in merged.items():
                task_name = f"{base_task}+{ep}"
                log.eval.task = task_name
                log.eval.task_display_name = task_name
                # Pilot filename shape: <ts>_<major>|<minor>-<epoch>_<eval_id>.eval
                ts = log.eval.created.replace(":", "-").replace("+00-00", "-00-00")
                fname = f"{ts}_{major}|{minor}-{ep}_{log.eval.eval_id}.eval"
                write_eval_log(log, str(out_dir / fname))
                header = read_eval_log(str(out_dir / fname), header_only=True)
                logs_manifest[fname] = json.loads(header.model_dump_json(exclude_none=True))
                tasks.append({
                    "name": task_name, "task_id": log.eval.task_id,
                    "task_args": log.eval.task_args or {},
                    "model": log.eval.model, "model_args": log.eval.model_args or {},
                    "sequence": ep,
                })
                total_logs += 1
            set_id = next(iter(merged.values())).eval.eval_set_id or ""
            (out_dir / ".eval-set-id").write_text(set_id)
            (out_dir / "eval-set.json").write_text(
                json.dumps({"eval_set_id": set_id, "tasks": tasks}, indent=2))
            (out_dir / "logs.json").write_text(json.dumps(logs_manifest, indent=2))
        print(f"{judge}: {len(pairs)} pairs")
    print(f"wrote {total_logs} merged logs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
