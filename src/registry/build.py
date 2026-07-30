"""Build a regenerable metadata index over the raw eval-log heap."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from inspect_ai.log import read_eval_log
import pandas as pd
import zipfile_zstd  # noqa: F401 - registers zstd support with zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_REGISTRY = DEFAULT_DATA_DIR / "registry.csv"

REGISTRY_COLUMNS = [
    "path",
    "eval_id",
    "status",
    "model",
    "benchmark",
    "dataset",
    "task_args",
    "created",
    "samples",
    "mtime",
]


def build_registry(
    data_dir: Path = DEFAULT_DATA_DIR,
    output: Path = DEFAULT_REGISTRY,
) -> tuple[pd.DataFrame, list[tuple[Path, str]]]:
    """Index Inspect headers without relying on the layout beneath ``data/``."""
    rows = []
    errors = []
    for path in sorted(data_dir.rglob("*.eval")):
        try:
            log = read_eval_log(path, header_only=True)
            info = log.eval
            rows.append(
                {
                    "path": _portable_path(path),
                    "eval_id": info.eval_id,
                    "status": str(log.status),
                    "model": str(info.model),
                    "benchmark": str(info.task),
                    "dataset": getattr(info.dataset, "name", None),
                    "task_args": json.dumps(
                        info.task_args or {}, sort_keys=True, default=str
                    ),
                    "created": info.created,
                    "samples": getattr(info.dataset, "samples", None),
                    "mtime": path.stat().st_mtime,
                }
            )
        except Exception as exc:
            errors.append((path, str(exc)))

    frame = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    return frame, errors


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    frame, errors = build_registry()
    print(f"Indexed {len(frame)} eval logs in {DEFAULT_REGISTRY}")
    for path, error in errors:
        print(f"warning: could not index {path}: {error}", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
