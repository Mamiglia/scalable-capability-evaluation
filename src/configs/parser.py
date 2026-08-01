"""CLI argument parsing utilities for task args, model args, and eval flags."""

import argparse
from typing import Any


def coerce_value(val: str) -> Any:
    """Coerce string values to int, float, bool, or original string."""
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def kv_pair(s: str) -> tuple[str, Any]:
    """Parse a KEY=VALUE string into a (key, value) tuple."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"Invalid KEY=VALUE pair: {s}")
    key, val = s.split("=", 1)
    return key.strip(), coerce_value(val.strip())


def flag_parse(extra: list[str] | str) -> dict[str, Any]:
    """Parse a list or space-separated string of flags into a dictionary."""
    is_flag = lambda x: x.startswith("-")
    res: dict[str, Any] = {}

    vals = extra.split(" ") if isinstance(extra, str) else list(extra)
    while vals and vals[0].startswith("-"):
        flag = vals.pop(0)
        assert is_flag(flag), f"Invalid flag: {flag}"
        flag = flag.lstrip("-").replace("-", "_")
        if vals and not is_flag(vals[0]):
            res[flag] = coerce_value(vals.pop(0))
        else:
            res[flag] = True
    return res
