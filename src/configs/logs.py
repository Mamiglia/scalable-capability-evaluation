"""One shared logger: INFO to stdout, plus ``<log_dir>/run.log`` once ``log_to`` is called."""

import logging
import sys
from pathlib import Path

FORMATTER = logging.Formatter("[%(levelname)s] %(asctime)s | %(filename)s | %(message)s", "%H:%M")



class _Stdout:
    """Resolves sys.stdout at write time, so a child that redirects stdout (collect.runner) takes the log with it."""

    def write(self, text: str) -> None:
        sys.stdout.write(text)

    def flush(self) -> None:
        sys.stdout.flush()


logger = logging.getLogger("small_judge")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(_Stdout()))
logger.handlers[0].setFormatter(FORMATTER)


def log_to(log_dir: Path | str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / "run.log")
    handler.setFormatter(FORMATTER)
    logger.addHandler(handler)
