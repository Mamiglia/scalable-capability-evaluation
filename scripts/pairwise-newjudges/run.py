"""Run one judging-variation config against a local (or any OpenAI-compatible) judge.

Resume: pair dirs holding a success log are skipped, so re-running only does new work.

Usage:
  uv run python run.py --config configs/gpqa_a_baseline.yaml \
      --judge-model openai-api/local/judge --judge-short qwen3-0.6b [--dry-run]
"""
import argparse
import json
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  epoch .eval zips use zstd; py<3.14 zipfile needs this shim
except ImportError:
    pass

import yaml
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log

from judge.task import judge_task

# configs hold repo-root-relative paths so they work locally and on the cluster
ROOT = Path(__file__).resolve().parents[2]


def has_success_log(directory: Path) -> bool:
    for p in directory.glob("*.eval"):
        try:
            if read_eval_log(str(p), header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


def validated_truncate_keep(config: dict) -> str:
    """Read truncate_keep from config; only "first" or "last" is a valid value."""
    keep = config.get("truncate_keep", "first")
    if keep not in ("first", "last"):
        raise ValueError(f"truncate_keep must be 'first' or 'last', got {keep!r}")
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-short", required=True)
    ap.add_argument("--base-url", default="", help="empty = provider default (API judges)")
    ap.add_argument("--model-args", default="", help="JSON model_args, e.g. openrouter provider pins")
    ap.add_argument("--limit", type=int, default=0, help="cap items per pair (pilot)")
    ap.add_argument("--max-connections", type=int, default=32)
    ap.add_argument("--log-root", default="", help="override config log_root (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    log_root = Path(args.log_root) if args.log_root else ROOT / cfg["log_root"]
    # truncate_fracs: list of fractions, each a length-control run with both
    # transcripts cut to that fraction of the shorter one (logs under
    # trunc-<pct>/, or trunc-end-<pct>/ if truncate_keep: last).
    fracs = cfg.get("truncate_fracs") or [None]
    keep = validated_truncate_keep(cfg)

    pending = []
    for frac in fracs:
        if frac is None:
            root, suffix = log_root, ""
        elif keep == "last":
            root, suffix = log_root / f"trunc-end-{int(frac * 100)}", f"_tend{int(frac * 100)}"
        else:
            root, suffix = log_root / f"trunc-{int(frac * 100)}", f"_t{int(frac * 100)}"
        for major, minor in cfg["comparisons"]:
            pair_dir = root / args.judge_short / f"{major}-vs-{minor}"
            if has_success_log(pair_dir):
                continue
            base_ids = cfg["sample_ids"][f"{major}-vs-{minor}"]
            if not base_ids:
                continue  # empty id list must not fall through to "all samples"
            if args.limit:
                base_ids = base_ids[:args.limit]
            # judge dataset ids are <item>_1 (normal) and <item>_2 (reversed)
            sample_ids = [f"{i}_{t}" for i in base_ids for t in (1, 2)]
            task = judge_task(
                major_path=ROOT / cfg["transcripts"][major],
                minor_path=ROOT / cfg["transcripts"][minor],
                system_prompt=cfg["system_prompt"],
                max_length=cfg["max_length"],
                include_reasoning=cfg["include_reasoning"],
                truncate_frac=frac,
                truncate_keep=keep,
                task_name=f"{cfg['condition']}_{args.judge_short}_{major}-vs-{minor}{suffix}",
                experiment_name=args.config.stem,
            )
            if cfg.get("invert_at_analysis"):
                # the dataset template ends every prompt with "...more capable
                # model?" regardless of system prompt; reword it for this condition
                for s in task.dataset:
                    s.input = s.input.replace(
                        "is from the more capable model? Answer",
                        "is from the less capable model? Answer")
            pending.append((major, minor, task, sample_ids, pair_dir))

    n_total = len(fracs) * len(cfg["comparisons"])
    print(f"{len(pending)} pair-runs pending (of {n_total})")
    if args.dry_run:
        for major, minor, task, ids, _ in pending:
            print(f"  {major}-vs-{minor}: {len(ids)} samples")
        return

    # fail_on_error: boundary items can overflow the judge context (400s);
    # tolerate up to 20% per pair. A failed pair skips, the job continues -
    # it stays resumable and one bad pair no longer kills the whole night.
    failed = []
    for major, minor, task, ids, pair_dir in pending:
        success, _ = eval_set(
            tasks=[task], model=args.judge_model,
            model_base_url=args.base_url or None,
            model_args=json.loads(args.model_args) if args.model_args else {},
            log_dir=str(pair_dir), sample_id=ids,
            max_connections=args.max_connections,
            fail_on_error=0.2,
        )
        if not success:
            print(f"FAILED (continuing): {major}-vs-{minor}")
            failed.append(f"{major}-vs-{minor}")
    if failed:
        raise SystemExit(f"{len(failed)} pair(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
