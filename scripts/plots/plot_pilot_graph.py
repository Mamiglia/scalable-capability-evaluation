#!/usr/bin/env python3
"""Judged models as nodes, judge accuracy on each pair as directed edges.

Arrow points to the model judges preferred: toward the truly better model when
pair accuracy > 50%, inverted when judges get the pair wrong more often.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

INK = "#17202A"
RIGHT = "#1baf7a"
WRONG = "#eb6834"


def short(name: str) -> str:
    return name.rsplit("/", 1)[-1].replace("claude-", "").replace("-it", "")


def pair_accuracy(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Per-pair accuracy, averaged over judges (judge is the replication unit)."""
    valid = verdicts[(verdicts.condition == "reasoning_off") & verdicts.correct.notna()]
    cells = valid.groupby(["benchmark", "pair", "judge"]).correct.mean()
    rows = cells.groupby(["benchmark", "pair"]).agg(accuracy="mean", judges="size").reset_index()
    better = valid.groupby(["benchmark", "pair"]).agg(
        better=("better_model", "first"), major=("major", "first"), minor=("minor", "first")
    )
    rows = rows.join(better, on=["benchmark", "pair"])
    rows["worse"] = rows.apply(lambda r: r.minor if r.better == r.major else r.major, axis=1)
    return rows


def spring_positions(rows: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Spring layout per connected component, components packed side by side."""
    graph = nx.Graph((r.better, r.worse) for r in rows.itertuples())
    positions = {}
    left = 0.0
    for component in sorted(nx.connected_components(graph), key=len, reverse=True):
        sub = graph.subgraph(component)
        pos = nx.spring_layout(sub, seed=seed, k=1.3, iterations=600)
        coords = np.array(list(pos.values()))
        coords = (coords - coords.min(axis=0)) / np.maximum(np.ptp(coords, axis=0), 1e-9)
        scale = math.sqrt(len(component) / graph.number_of_nodes())
        for node, xy in zip(pos, coords * scale):
            positions[node] = np.array([left + xy[0], xy[1] - scale / 2])
        left += scale + .25
    return positions


OFFSETS = [(0, 1), (1, .4), (-1, .4), (0, -1), (1, -.6), (-1, -.6), (1.6, 0), (-1.6, 0)]


def place_labels(ax: plt.Axes, pos: dict[str, np.ndarray], pad: float = 3.5) -> None:
    """Greedy label placement: first candidate offset that clears the earlier labels."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    taken = [ax.transData.transform(xy) for xy in pos.values()]
    boxes = []
    for model in sorted(pos, key=lambda m: (pos[m][1], pos[m][0])):
        for i, (dx, dy) in enumerate(OFFSETS):
            text = ax.annotate(
                short(model), pos[model], xytext=(pad * dx, pad * dy), textcoords="offset points",
                fontsize=4.5, color=INK, zorder=4, ha="left" if dx > 0 else "right" if dx < 0 else "center",
                va="bottom" if dy > 0 else "top" if dy < 0 else "center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": .4})
            box = text.get_window_extent(renderer).expanded(1.02, 1.15)
            clear = not any(box.overlaps(other) for other in boxes) and not any(
                box.contains(*point) for point in taken)
            if clear or i == len(OFFSETS) - 1:
                boxes.append(box)
                break
            text.remove()


def draw(rows: pd.DataFrame, out: Path, seed: int) -> None:
    """Node-link graph: models are nodes, judged pairs are directed edges."""
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 7, "legend.fontsize": 6,
                         "svg.fonttype": "none"})
    pos = spring_positions(rows, seed)
    label_axes = []
    benches = sorted(rows.benchmark.unique())
    fig, axes = plt.subplots(len(benches), 1, figsize=(3.4, 3.9), layout="constrained")
    for ax, bench in zip(axes, benches):
        for r in rows[rows.benchmark == bench].itertuples():
            source, target = (r.worse, r.better) if r.accuracy > .5 else (r.better, r.worse)
            ax.add_patch(FancyArrowPatch(
                pos[source], pos[target], connectionstyle="arc3,rad=.12", arrowstyle="-|>",
                mutation_scale=5, shrinkA=2, shrinkB=2, linewidth=abs(r.accuracy - .5) * 5,
                color=RIGHT if r.accuracy > .5 else WRONG, alpha=.8, zorder=2))
        points = np.array([pos[m] for m in pos])
        ax.scatter(points[:, 0], points[:, 1], s=4, color=INK, zorder=3)
        label_axes.append(ax)
        ax.set_title(bench, loc="left", fontsize=7, color="#666666")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.margins(.12)
    fig.set_layout_engine("none")  # freeze axes before measuring text boxes
    for ax in label_axes:
        place_labels(ax, pos)
    handles = [plt.Line2D([], [], color=RIGHT, lw=abs(a - .5) * 5, label=f"{a:.0%}")
               for a in (.6, .8, 1.)]
    handles.append(plt.Line2D([], [], color=WRONG, lw=1.5, label="< 50% (arrow inverted)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               handlelength=1.4, columnspacing=1., title="Judge accuracy on the pair",
               title_fontsize=6)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def self_check() -> None:
    toy = pd.DataFrame([
        {"condition": "reasoning_off", "benchmark": "GPQA", "pair": "a vs b", "judge": j,
         "correct": c, "better_model": "b", "major": "a", "minor": "b"}
        for j, c in (("j1", 1.0), ("j2", 1.0))
    ])
    row = pair_accuracy(toy).iloc[0]
    assert row.accuracy == 1 and row.better == "b" and row.worse == "a"
    print("self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts", type=Path, default=Path("analysis/pilot-results/verdicts.csv"))
    parser.add_argument("--out", type=Path, default=Path("analysis/pilot-results/10_judge_graph"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    verdicts = pd.read_csv(args.verdicts, low_memory=False)
    rows = pair_accuracy(verdicts)
    draw(rows, args.out, args.seed)
    rows.to_csv(args.out.with_name(args.out.name + "_edges.csv"), index=False)
    print(f"wrote {args.out}.png / .svg ({len(rows)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
