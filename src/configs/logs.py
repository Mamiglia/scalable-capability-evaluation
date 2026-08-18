"""One shared logger: INFO to stdout, plus ``<log_dir>/run.log`` once ``log_to`` is called."""

import logging
import sys
from pathlib import Path

FORMATTER = logging.Formatter("[%(levelname)s] %(asctime)s | %(filename)s | %(message)s", "%H:%M")

logger = logging.getLogger("small_judge")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.handlers[0].setFormatter(FORMATTER)


def log_to(log_dir: Path | str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / "run.log")
    handler.setFormatter(FORMATTER)
    logger.addHandler(handler)
