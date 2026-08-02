#!/usr/bin/env python3
"""
Split Inspect evaluation log files (.eval) containing multiple epochs
into individual single-epoch evaluation log files (.eval), preserving and handling sample.id.

Usage examples:
    # Dry-run split scan on a directory
    python scripts/split_eval_epochs.py data/gpqa --dry-run

    # In-place split multi-epoch files into 0.eval, 1.eval, 2.eval, ... in the same directory
    python scripts/split_eval_epochs.py data/gpqa --inplace

    # Output split files to a custom directory
    python scripts/split_eval_epochs.py data/gpqa -o output_dir/
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Ensure src/ is on sys.path so configs can be imported if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inspect_ai.log import EvalLog, read_eval_log, write_eval_log


def group_samples_by_epoch(
    samples: List[Any],
    sort_by_id: bool = False,
) -> Dict[int, List[Any]]:
    """
    Group samples by epoch, maintaining sample.id integrity.

    Returns:
        Dict mapping epoch index (1, 2, ...) to list of sample objects.
    """
    if not samples:
        return {}

    # Strategy 1: Check sample.epoch attribute
    epochs_present = {getattr(s, "epoch", None) for s in samples}
    if len(epochs_present) > 1 and None not in epochs_present:
        by_epoch: Dict[int, List[Any]] = {}
        valid = True
        for s in samples:
            ep = getattr(s, "epoch")
            by_epoch.setdefault(ep, []).append(s)

        # Verify that within each epoch, sample IDs are unique
        for ep, ep_samples in by_epoch.items():
            ids = [getattr(s, "id", None) for s in ep_samples]
            if len(ids) != len(set(ids)):
                valid = False
                break

        if valid:
            # Remap 1-indexed contiguous epoch keys if needed
            sorted_epochs = sorted(by_epoch.keys())
            remapped = {i + 1: by_epoch[ep] for i, ep in enumerate(sorted_epochs)}
            if sort_by_id:
                for ep in remapped:
                    remapped[ep] = sorted(remapped[ep], key=lambda s: str(getattr(s, "id", "")))
            return remapped

    # Strategy 2: Occurrence-based grouping by sample.id
    by_epoch_occurrence: Dict[int, List[Any]] = {}
    id_counts: Dict[Any, int] = {}

    for s in samples:
        s_id = getattr(s, "id", None)
        count = id_counts.get(s_id, 0) + 1
        id_counts[s_id] = count
        by_epoch_occurrence.setdefault(count, []).append(s)

    if sort_by_id:
        for ep in by_epoch_occurrence:
            by_epoch_occurrence[ep] = sorted(
                by_epoch_occurrence[ep], key=lambda s: str(getattr(s, "id", ""))
            )

    return by_epoch_occurrence


def split_eval_log(
    log: EvalLog,
    sort_by_id: bool = False,
    reset_sample_epoch: bool = True,
) -> Dict[int, EvalLog]:
    """
    Split a multi-epoch EvalLog into a dictionary of single-epoch EvalLog objects.

    Args:
        log: The source EvalLog object.
        sort_by_id: Whether to sort samples by sample.id in each epoch log.
        reset_sample_epoch: Whether to reset sample.epoch to 1 in split single-epoch logs.

    Returns:
        Dict mapping epoch index (1, 2, ...) to split EvalLog object.
    """
    if not log.samples:
        return {1: log}

    grouped_samples = group_samples_by_epoch(log.samples, sort_by_id=sort_by_id)

    # If only 1 epoch is present, return original log
    if len(grouped_samples) <= 1:
        return {1: log}

    split_logs: Dict[int, EvalLog] = {}

    for epoch_num, epoch_samples in grouped_samples.items():
        # Clone parent log
        split_log = log.model_copy(deep=True)

        # Optionally reset sample.epoch to 1 for standalone single-epoch logs
        if reset_sample_epoch:
            prepared_samples = []
            for s in epoch_samples:
                s_copy = s.model_copy(deep=True) if hasattr(s, "model_copy") else s
                if hasattr(s_copy, "epoch"):
                    s_copy.epoch = 1
                prepared_samples.append(s_copy)
            split_log.samples = prepared_samples
        else:
            split_log.samples = epoch_samples

        # Update results summary
        if split_log.results:
            split_log.results.total_samples = len(epoch_samples)
            split_log.results.completed_samples = sum(
                1 for s in epoch_samples if not getattr(s, "error", None)
            )

        # Update config epochs count
        if hasattr(split_log, "eval") and split_log.eval and hasattr(split_log.eval, "config") and split_log.eval.config:
            if hasattr(split_log.eval.config, "epochs"):
                split_log.eval.config.epochs = 1

        split_logs[epoch_num] = split_log

    return split_logs


def process_eval_file(
    file_path: Path,
    output_dir: Optional[Path] = None,
    inplace: bool = False,
    sort_by_id: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[bool, List[Path]]:
    """
    Process a single .eval file, splitting it if it contains multiple epochs.

    Returns:
        (was_split, list_of_output_paths)
    """
    try:
        log = read_eval_log(file_path, header_only=False)
    except Exception as e:
        if verbose:
            print(f"Error reading {file_path}: {e}")
        return False, []

    if not log.samples:
        if verbose:
            print(f"Skipping {file_path}: No samples found.")
        return False, []

    split_logs = split_eval_log(log, sort_by_id=sort_by_id)
    num_epochs = len(split_logs)

    if num_epochs <= 1:
        if verbose:
            print(f"Skipping {file_path}: Single epoch file ({len(log.samples)} samples).")
        return False, []

    output_paths: List[Path] = []
    parent_dir = file_path.parent

    # Determine destination naming scheme
    for ep_num, split_log in split_logs.items():
        ep_idx = ep_num - 1  # 0-indexed for filename numbering (0.eval, 1.eval, ...)

        if output_dir:
            dest_dir = output_dir / parent_dir.name if inplace else output_dir
            dest_file = dest_dir / f"{file_path.stem}_epoch{ep_num}.eval"
        elif inplace:
            dest_file = parent_dir / f"{ep_idx}.eval"
        else:
            dest_file = parent_dir / f"{file_path.stem}_epoch{ep_num}.eval"

        output_paths.append(dest_file)

    print(f"Found {num_epochs} epochs in {file_path} ({len(log.samples)} total samples):")
    for ep_num, dest_file in zip(split_logs.keys(), output_paths):
        sample_count = len(split_logs[ep_num].samples or [])
        action_str = "[DRY-RUN] Would write" if dry_run else "Writing"
        print(f"  - {action_str} Epoch {ep_num} ({sample_count} samples) -> {dest_file}")

    if not dry_run:
        # Write to temporary paths first for safety in case of inplace operations
        written_files = []
        for ep_num, dest_file in zip(split_logs.keys(), output_paths):
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(split_logs[ep_num], dest_file)
            written_files.append(dest_file)

        # If inplace and original file is not in output_paths, remove old multi-epoch source file
        if inplace and file_path not in written_files:
            file_path.unlink(missing_ok=True)

    return True, output_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split multi-epoch Inspect evaluation log files (.eval) into single-epoch files."
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="Input folder(s) or .eval file(s) to process.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Target base directory for split eval files.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Split in-place in source directory using sequential indexing (0.eval, 1.eval, ...).",
    )
    parser.add_argument(
        "--sort-by-id",
        action="store_true",
        help="Sort samples by sample.id in each split epoch file.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing or modifying files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose details during execution.",
    )

    args = parser.parse_args()

    eval_files: List[Path] = []
    for ip in args.input_paths:
        if ip.is_file() and ip.suffix == ".eval":
            eval_files.append(ip)
        elif ip.is_dir():
            eval_files.extend(sorted(ip.rglob("*.eval")))

    if not eval_files:
        print("No .eval files found in input paths.")
        return 0

    print(f"Scanning {len(eval_files)} .eval file(s)...")

    split_count = 0
    total_created = 0

    for file_path in eval_files:
        was_split, created = process_eval_file(
            file_path=file_path,
            output_dir=args.output_dir,
            inplace=args.inplace,
            sort_by_id=args.sort_by_id,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if was_split:
            split_count += 1
            total_created += len(created)

    print("\n--- Summary ---")
    print(f"Total .eval files scanned : {len(eval_files)}")
    print(f"Multi-epoch files split   : {split_count}")
    print(f"Single-epoch files created : {total_created}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
