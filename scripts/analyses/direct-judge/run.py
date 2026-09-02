# scripts/analyses/direct-judge/run.py
"""Run one direct-judge config against a local (or any OpenAI-compatible) judge.

Resume: model dirs holding a success log are skipped, so re-running only does new work.

Usage:
  uv run python run.py --config configs/gpqa.yaml \
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

from score_task import score_task

# configs hold repo-root-relative paths so they work locally and on the cluster
ROOT = Path(__file__).resolve().parents[3]


def has_success_log(directory: Path) -> bool:
    for p in directory.glob("*.eval"):
        try:
            if read_eval_log(str(p), header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-short", required=True)
    ap.add_argument("--base-url", default="", help="empty = provider default (API judges)")
    ap.add_argument("--model-args", default="", help="JSON model_args, e.g. openrouter provider pins")
    ap.add_argument("--limit", type=int, default=0, help="cap items per model (pilot)")
    ap.add_argument("--max-connections", type=int, default=32)
    ap.add_argument("--log-root", default="", help="override config log_root (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    log_root = Path(args.log_root) if args.log_root else ROOT / cfg["log_root"]
    pending = []
    for short in cfg["sample_ids"]:
        model_dir = log_root / args.judge_short / short
        if has_success_log(model_dir):
            continue
        ids = cfg["sample_ids"][short]
        if not ids:
            continue
        if args.limit:
            ids = ids[:args.limit]
        task = score_task(
            log_path=ROOT / cfg["transcripts"][short],
            item_ids=ids,
            system_prompt=cfg["system_prompt"],
            max_length=cfg["max_length"],
            include_reasoning=cfg["include_reasoning"],
            task_name=f"score_{args.judge_short}_{short}",
            experiment_name=args.config.stem,
        )
        pending.append((short, task, ids, model_dir))

    print(f"{len(pending)} model-runs pending (of {len(cfg['sample_ids'])})")
    if args.dry_run:
        for short, task, ids, _ in pending:
            print(f"  {short}: {len(ids)} transcripts")
        return

    # fail_on_error: boundary items can overflow the judge context (400s);
    # tolerate up to 20% per model. A failed model skips, the job continues.
    failed = []
    for short, task, ids, model_dir in pending:
        success, _ = eval_set(
            tasks=[task], model=args.judge_model,
            model_base_url=args.base_url or None,
            model_args=json.loads(args.model_args) if args.model_args else {},
            log_dir=str(model_dir),
            max_connections=args.max_connections,
            fail_on_error=0.2,
        )
        if not success:
            print(f"FAILED (continuing): {short}")
            failed.append(short)
    if failed:
        raise SystemExit(f"{len(failed)} model(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
