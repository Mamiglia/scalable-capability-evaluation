"""House-style figure primitives (Arial, 400 dpi, constrained layout)."""

import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import wbconfig

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#33322f", "#4a4844", "#d3d2c9", "#ffffff"
PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#8c6d31", "#0b7285"
]
JRAMP = LinearSegmentedColormap.from_list("jramp", ["#b23c33", "#b8b5ab", "#1f63b6"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#bcd4f0", "#2a78d6", "#12365e"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#2a78d6", "#f6f4ee", "#e34948"])

COL, WIDE = 3.5, 7.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "lines.linewidth": 1.6,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def new_fig(w, h):
    """Constrained-layout figure at size (w, h) inches."""
    return plt.figure(figsize=(w, h), layout="constrained")


def save(fig, stem):
    """Write {stem}.png and {stem}.svg to out/figures/; close figure."""
    out_dir = pathlib.Path(wbconfig.cfg()["paths"]["out"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(str(out_dir / f"{stem}.{ext}"), facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
