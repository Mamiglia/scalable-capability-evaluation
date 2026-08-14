#!/usr/bin/env python3
"""
Digest existing Inspect evaluation log files (.eval), filtering for success,
completeness, and deduplication, and moving (or copying) them into the expected
repository directory structure under data/:

    data/{benchmark}/{task_args_hash}/{model_shortname}/{index}/{inspect_id}.eval
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src/ is on sys.path so configs can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inspect_ai.log import EvalLog, read_eval_log
from configs import TaskConfig, ModelConfig
from configs.paths import get_eval_path


def get_log_identifier(source_path: Path, log: Optional[EvalLog] = None) -> str:
    """
    Get a unique identifier for an eval log:
    Prefers log.eval.eval_id if available, otherwise SHA256 hash of the file.
    """
    if log and hasattr(log, "eval") and getattr(log.eval, "eval_id", None):
        return f"eval_id:{log.eval.eval_id}"
    
    # Fallback to file content SHA256
    file_bytes = source_path.read_bytes()
    h = hashlib.sha256(file_bytes).hexdigest()
    return f"sha256:{h}"


def is_successful_and_complete(log: EvalLog) -> Tuple[bool, str]:
    """
    Check if an evaluation log is successful and complete.

    Returns:
        (is_valid, reason) tuple.
    """
    if not hasattr(log, "status") or log.status != "success":
        status_str = getattr(log, "status", "unknown")
        return False, f"Status is '{status_str}' (expected 'success')"

    if not hasattr(log, "results") or log.results is None:
        return False, "Missing results"

    total = getattr(log.results, "total_samples", None)
    completed = getattr(log.results, "completed_samples", None)

    if total is None or total <= 0:
        return False, f"Invalid total samples: {total}"

    if completed is None or completed < total:
        return False, f"Incomplete samples: completed {completed}/{total}"

    if hasattr(log, "samples") and log.samples is not None:
        if len(log.samples) != total:
            return False, f"Sample count mismatch: {len(log.samples)} samples vs {total} expected"

    return True, "OK"


def get_target_path(
    log: EvalLog,
    base_dir: Path,
    source_path: Path,
    preserve_names: bool = False,
    used_targets: Optional[Set[Path]] = None,
) -> Optional[Path]:
    """
    Compute the destination path for an evaluation log based on task and model configs.
    """
    eval_info = getattr(log, "eval", None)
    if not eval_info:
        return None

    task_name = getattr(eval_info, "task", None)
    model_name = getattr(eval_info, "model", None)

    if not task_name or not model_name:
        return None

    task_args = getattr(eval_info, "task_args", None) or {}

    task_cfg = TaskConfig(name=task_name, task_args=task_args)
    model_cfg = ModelConfig(name=model_name)

    dest_dir = get_eval_path(task=task_cfg, model=model_cfg, base_dir=base_dir)

    eval_id = getattr(eval_info, "eval_id", None)
    file_name = f"{eval_id}.eval" if eval_id else source_path.name

    if preserve_names:
        dest_file = dest_dir / source_path.name
        if used_targets is not None and dest_file in used_targets:
            counter = 1
            stem = source_path.stem
            ext = source_path.suffix
            while (dest_dir / f"{stem}_{counter}{ext}") in used_targets:
                counter += 1
            dest_file = dest_dir / f"{stem}_{counter}{ext}"
        return dest_file

    # Sequential epoch directory naming: 0/{eval_id}.eval, 1/{eval_id}.eval, ...
    if source_path.parent.name.isdigit():
        i = int(source_path.parent.name)
    elif source_path.stem.isdigit():
        i = int(source_path.stem)
    else:
        i = 0

    while True:
        candidate_dir = dest_dir / str(i)
        candidate = candidate_dir / file_name
        if used_targets is not None and candidate in used_targets:
            i += 1
            continue
        if candidate.exists() and candidate.resolve() != source_path.resolve():
            i += 1
            continue
        if candidate_dir.exists() and any(candidate_dir.glob("*.eval")) and not (candidate.exists() and candidate.resolve() == source_path.resolve()):
            i += 1
            continue
        return candidate


def index_existing_target_logs(target_dir: Path) -> Set[str]:
    """
    Scan target_dir for existing .eval logs and return a set of their unique identifiers.
    """
    seen_ids: Set[str] = set()
    if not target_dir.exists():
        return seen_ids

    for existing_file in target_dir.rglob("*.eval"):
        try:
            log = read_eval_log(existing_file, header_only=True)
            ident = get_log_identifier(existing_file, log)
            seen_ids.add(ident)
        except Exception:
            # If an existing file cannot be read, index its file hash
            try:
                ident = get_log_identifier(existing_file)
                seen_ids.add(ident)
            except Exception:
                pass
    return seen_ids


def digest_evals(
    input_paths: List[Path],
    target_dir: Path = Path("data"),
    copy_files: bool = False,
    preserve_names: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Find, filter, deduplicate, and digest .eval files from input paths into target_dir.
    """
    eval_files: List[Path] = []
    for ip in input_paths:
        if ip.is_file() and ip.suffix == ".eval":
            eval_files.append(ip)
        elif ip.is_dir():
            eval_files.extend(sorted(ip.rglob("*.eval")))

    stats = {
        "found": len(eval_files),
        "processed": 0,
        "skipped": 0,
        "reasons": {},
    }

    # Index existing target directory logs to prevent duplicate digestion across runs
    seen_identifiers: Set[str] = index_existing_target_logs(target_dir)
    used_targets: Set[Path] = set()

    for src_path in eval_files:
        try:
            log = read_eval_log(src_path, header_only=False)
        except Exception as e:
            reason = f"Read error / corrupted file: {e}"
            stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if verbose:
                print(f"Skipping {src_path}: {reason}")
            continue

        # 1. Filter by success and completeness
        valid, reason = is_successful_and_complete(log)
        if not valid:
            stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if verbose:
                print(f"Skipping {src_path}: {reason}")
            continue

        # 2. Check for duplicates
        ident = get_log_identifier(src_path, log)
        if ident in seen_identifiers:
            reason = "Duplicate evaluation log (already processed or exists in target)"
            stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if verbose:
                print(f"Skipping {src_path}: {reason}")
            continue

        # 3. Determine target path
        dest_file = get_target_path(
            log=log,
            base_dir=target_dir,
            source_path=src_path,
            preserve_names=preserve_names,
            used_targets=used_targets,
        )

        if not dest_file:
            reason = "Missing task or model metadata in eval log"
            stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if verbose:
                print(f"Skipping {src_path}: {reason}")
            continue

        if src_path.resolve() == dest_file.resolve():
            reason = "File already at target path"
            stats["skipped"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            if verbose:
                print(f"Skipping {src_path}: {reason}")
            continue

        seen_identifiers.add(ident)
        used_targets.add(dest_file)
        action_str = "Copying" if copy_files else "Moving"

        if dry_run:
            print(f"[DRY-RUN] {action_str} {src_path} -> {dest_file}")
        else:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if copy_files:
                shutil.copy2(src_path, dest_file)
            else:
                shutil.move(src_path, dest_file)
            print(f"{action_str} {src_path} -> {dest_file}")

        stats["processed"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Digest Inspect evaluation log folders, filter for success, completeness, & deduplication, and move/copy into repository data structure."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="Input folder(s) or file(s) containing Inspect .eval logs.",
    )
    parser.add_argument(
        "-t",
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Target base directory for digested evals (default: data).",
    )
    parser.add_argument(
        "-c",
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--preserve-names",
        action="store_true",
        help="Preserve original filenames instead of sequential 0.eval, 1.eval, ... naming.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Perform a dry run without moving or copying files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose details for skipped files.",
    )

    args = parser.parse_args()

    print(f"Scanning {len(args.input_paths)} path(s)...")
    stats = digest_evals(
        input_paths=args.input_paths,
        target_dir=args.target_dir,
        copy_files=args.copy,
        preserve_names=args.preserve_names,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print("\n--- Summary ---")
    print(f"Total .eval files found : {stats['found']}")
    print(f"Files processed          : {stats['processed']}")
    print(f"Files skipped            : {stats['skipped']}")
    if stats["reasons"]:
        print("Skip reasons:")
        for reason, count in stats["reasons"].items():
            print(f"  - {reason}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
