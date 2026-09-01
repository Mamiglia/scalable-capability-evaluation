from pathlib import Path

from inspect_ai.log import read_eval_log


def has_success_log(directory: Path) -> bool:
    """Whether a directory contains a finished eval log."""
    for path in directory.glob("*.eval"):
        try:
            if read_eval_log(str(path), header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False
