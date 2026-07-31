"""Build soft-links in data/dataset/ pointing to raw eval logs in data/raw/."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Any

from inspect_ai.log import read_eval_log
import zipfile_zstd  # noqa: F401 - registers zstd support with zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "dataset"


def format_task_args(task_args: dict[str, Any] | None) -> str:
    if not task_args:
        return "default"
    items = []
    for k, v in sorted(task_args.items()):
        items.append(f"{k}={v}")
    return "_".join(items)


def _model_key(value: str) -> str:
    return str(value).rsplit("/", 1)[-1].casefold()


def _benchmark_key(value: str) -> str:
    return str(value).rsplit("/", 1)[-1].casefold().replace("-", "_")


def build_registry(
    data_dir: Path = DEFAULT_DATA_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> tuple[dict[tuple[str, str, str], list[Path]], list[tuple[Path, str]]]:
    """Index raw Inspect logs and soft-link them under ``data/dataset/``."""
    data_dir = Path(data_dir)
    dataset_dir = Path(dataset_dir)
    raw_base = data_dir / "raw" if (data_dir / "raw").is_dir() else data_dir

    grouped_logs: dict[tuple[str, str, str], list[tuple[str, float, Path]]] = (
        defaultdict(list)
    )
    errors: list[tuple[Path, str]] = []

    # Find raw log files, excluding symlinks and any files inside dataset_dir
    all_eval_files = [
        p
        for p in data_dir.rglob("*.eval")
        if not p.is_symlink()
        and not _is_relative_to(p.absolute(), dataset_dir.absolute())
    ]

    for path in sorted(all_eval_files):
        try:
            log = read_eval_log(path, header_only=True)
            if str(log.status) != "success":
                continue
            info = log.eval

            # Extract benchmark from relative path under data/raw if available
            abs_path = path.resolve()
            if _is_relative_to(abs_path, raw_base.resolve()):
                rel_parent = abs_path.parent.relative_to(raw_base.resolve())
                bench_key = (
                    str(rel_parent)
                    if str(rel_parent) != "."
                    else _benchmark_key(info.task)
                )
            else:
                bench_key = _benchmark_key(info.task)

            model_key = _model_key(info.model)
            task_args_str = format_task_args(info.task_args)

            created = info.created or ""
            mtime = path.stat().st_mtime
            grouped_logs[(bench_key, task_args_str, model_key)].append(
                (created, mtime, abs_path)
            )
        except Exception as exc:
            errors.append((path, str(exc)))

    created_symlinks: set[Path] = set()
    result_map: dict[tuple[str, str, str], list[Path]] = {}

    for (bench_key, task_args_str, model_key), items in grouped_logs.items():
        # Sort by creation time, mtime, path
        items.sort(key=lambda x: (x[0], x[1], str(x[2])))
        target_dir = dataset_dir / bench_key / task_args_str / model_key
        target_dir.mkdir(parents=True, exist_ok=True)

        group_links: list[Path] = []
        for idx, (_, _, raw_path) in enumerate(items):
            link_name = f"{idx}.eval"
            symlink_path = target_dir / link_name

            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()

            rel_target = os.path.relpath(raw_path, symlink_path.parent)
            symlink_path.symlink_to(rel_target)

            created_symlinks.add(symlink_path.absolute())
            group_links.append(symlink_path)

        result_map[(bench_key, task_args_str, model_key)] = group_links

    # Clean up stale symlinks in dataset_dir
    if dataset_dir.exists():
        for symlink in dataset_dir.rglob("*.eval"):
            if symlink.is_symlink() and symlink.absolute() not in created_symlinks:
                symlink.unlink()

        _cleanup_empty_dirs(dataset_dir)

    return result_map, errors


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _cleanup_empty_dirs(dir_path: Path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in dirs:
            p = Path(root) / name
            try:
                p.rmdir()  # fails if non-empty
            except OSError:
                pass


def main() -> int:
    result_map, errors = build_registry()
    total_linked = sum(len(links) for links in result_map.values())
    print(f"Linked {total_linked} eval logs under {DEFAULT_DATASET_DIR}")
    for path, error in errors:
        print(f"warning: could not index {path}: {error}", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
