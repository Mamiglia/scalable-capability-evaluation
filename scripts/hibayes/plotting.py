"""Shared paths, rendering style, and JSON output for the GLM figures."""

import json
import math
import numbers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HIB = ROOT / "analysis" / "hibayes"
OUT = HIB / "figures"

LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
    s3="#1baf7a", s4="#eda100", s5="#e87ba4", s6="#008300",
    good="#0ca30c", crit="#d03b3b", warn="#fab219",
)
DARK = dict(
    surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
    s3="#199e70", s4="#c98500", s5="#d55181", s6="#008300",
    good="#0ca30c", crit="#d03b3b", warn="#fab219",
)


def style(colors: dict[str, str]) -> None:
    plt.rcParams.update({
        "figure.facecolor": colors["surface"],
        "axes.facecolor": colors["surface"],
        "savefig.facecolor": colors["surface"],
        "text.color": colors["ink"],
        "axes.labelcolor": colors["ink2"],
        "axes.edgecolor": colors["axis"],
        "xtick.color": colors["muted"],
        "ytick.color": colors["muted"],
        "grid.color": colors["grid"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
        "svg.fonttype": "none",
    })


def finish(fig, name: str, colors: dict[str, str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if colors is LIGHT else "_dark"
    fig.savefig(OUT / f"{name}{suffix}.png", dpi=150, bbox_inches="tight",
                facecolor=colors["surface"])
    if colors is LIGHT:
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight",
                    facecolor=colors["surface"])
    plt.close(fig)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_value(value), indent=1, allow_nan=False))
