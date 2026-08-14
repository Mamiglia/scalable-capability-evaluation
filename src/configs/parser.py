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


def _is_flag(token: str) -> bool:
    """A token is a flag if it starts with '-' and is not a negative number."""
    if not token.startswith("-"):
        return False
    return not token.lstrip("-").replace(".", "", 1).isdigit()


def flag_parse(extra: list[str] | str) -> dict[str, Any]:
    """Parse trailing CLI flags into a dictionary.

    Accepts both ``--flag value`` and ``--flag=value``; a flag with no value is True.
    """
    vals = extra.split(" ") if isinstance(extra, str) else list(extra)
    res: dict[str, Any] = {}

    while vals and _is_flag(vals[0]):
        flag = vals.pop(0).lstrip("-")
        if "=" in flag:
            flag, _, value = flag.partition("=")
            res[flag.replace("-", "_")] = coerce_value(value)
            continue
        flag = flag.replace("-", "_")
        if vals and not _is_flag(vals[0]):
            res[flag] = coerce_value(vals.pop(0))
        else:
            res[flag] = True
    if vals:
        raise ValueError(f"Unparsed CLI arguments: {vals}")
    return res
