"""House-style plot primitives for the judge analysis (Arial, 400 dpi).

Same design tokens as src/irt/plots.py and the lasr pilot; duplicated on
purpose while this lives in scripts/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#33322f", "#4a4844", "#d3d2c9", "#ffffff"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "lines.linewidth": 1.6,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def style(ax, grid_axis="y"):
    """Shared chrome: recessive grid, drop top/right spines, mute the rest."""
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def save(fig, out_stub):
    """Tight layout, write <out_stub>.png and .svg."""
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stub}.{ext}", facecolor=SURFACE)
    plt.close(fig)


def chance(ax, value=0.5, axis="y"):
    """Dashed reference line at chance."""
    line = ax.axhline if axis == "y" else ax.axvline
    line(value, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)


def legend(ax, loc="best", **kw):
    leg = ax.legend(frameon=False, loc=loc, handletextpad=0.4, labelspacing=0.3, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def dot_ci(ax, y, m, lo, hi, color, note=None, open_marker=False):
    """One horizontal CI whisker + dot at row y; optional right-hand note."""
    ax.plot([lo, hi], [y, y], color=color, lw=1.2, zorder=3, solid_capstyle="butt")
    face = SURFACE if open_marker else color
    ax.plot(m, y, "o", ms=5.5, mfc=face, mec=color, mew=1.2, zorder=4)
    if note:
        ax.annotate(note, xy=(1.01, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", color=MUTED, fontsize=6.5,
                    annotation_clip=False)
