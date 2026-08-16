"""Local checks for the cross-domain prep + analysis logic (no cluster, no API)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_cross_domain import pooled_eci, _item_held


def test_pooled_eci_zscore_average():
    logits = pd.DataFrame(
        {"gpqa_logit": [2.0, 0.0, -2.0],
         "aime_logit": [1.0, 0.0, -1.0],
         "chess_logit": [0.0, 0.0, 0.0]},   # zero-variance domain -> 0, must not NaN
        index=["a", "b", "c"])
    eci = pooled_eci(logits)
    assert eci.loc["a"] > eci.loc["b"] > eci.loc["c"]
    assert abs(eci.loc["b"]) < 1e-9
    assert eci.notna().all()


def test_item_held_deterministic_and_fraction():
    ids = [f"item-{i}" for i in range(4000)]
    frac = 0.2
    held = [_item_held(i, frac) for i in ids]
    assert held == [_item_held(i, frac) for i in ids]        # deterministic
    rate = sum(held) / len(held)
    assert abs(rate - frac) < 0.03                           # ~20% held
    # chess-style integer ids and gpqa-style rec ids both work, don't collide
    assert isinstance(_item_held(42, frac), bool)


def test_cell_accuracy_counterbalanced():
    from cross_domain_analysis import cell_accuracy
    # judge always right (picks slot with higher logit == label side)
    d = pd.DataFrame({"pair_uid": ["p", "p"], "order": [0, 1], "label": [1, 2],
                      "logit_1": [2.0, -2.0], "logit_2": [-2.0, 2.0]})
    assert cell_accuracy(d) == 1.0
    # judge always picks slot 1 -> right on order0 (label1), wrong on order1 (label2) -> 0.5
    d2 = pd.DataFrame({"pair_uid": ["p", "p"], "order": [0, 1], "label": [1, 2],
                       "logit_1": [2.0, 2.0], "logit_2": [-2.0, -2.0]})
    assert cell_accuracy(d2) == 0.5


def test_house_style_offsets_visible_axes():
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyses"))
    from plots import style

    fig, ax = plt.subplots()
    style(ax)
    assert ax.spines["left"].get_position() == ("outward", 5)
    assert ax.spines["bottom"].get_position() == ("outward", 5)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    plt.close(fig)


def _svg_width_points(path):
    import xml.etree.ElementTree as ET
    return float(ET.parse(path).getroot().attrib["width"].removesuffix("pt"))


def test_transfer_grid_is_compact_and_has_no_canvas_title(tmp_path):
    from cross_domain_analysis import figures

    rows = []
    for label in ("eci", "indomain"):
        for train in ("base", "gpqa", "aime", "chess", "loo_gpqa", "loo_aime", "loo_chess"):
            for test in ("gpqa", "aime", "chess"):
                rows.append(dict(label=label, train=train, test=test, n=10, acc=0.65))
    stub = figures(pd.DataFrame(rows), "Qwen3-8B", tmp_path)
    svg = Path(stub + ".svg")
    raw = svg.read_text()
    assert _svg_width_points(svg) <= 330
    assert "Cross-domain transfer: train-set x test-domain accuracy" not in raw


def test_gap_curves_are_compact_and_use_panel_tags(tmp_path):
    import json
    from cross_domain_analysis import gap_curves

    root, pairs_dir = tmp_path / "root", tmp_path / "pairs"
    pairs_dir.mkdir()
    for test in ("gpqa", "aime", "chess"):
        pair_rows, verdict_rows = [], []
        for i in range(12):
            for order in (0, 1):
                uid = f"{test}-{i}"
                pair_rows.append(dict(pair_uid=uid, order=order,
                                      cap_1=0.1 + i / 20, cap_2=0.1))
                hit = i % 3 != 0
                label = 1 if order == 0 else 2
                picked = label if hit else 3 - label
                verdict_rows.append(dict(pair_uid=uid, order=order, item=f"item-{i % 4}",
                                         label=label, logit_1=float(picked == 1),
                                         logit_2=float(picked == 2)))
        pd.DataFrame(pair_rows).to_parquet(pairs_dir / f"pairs_{test}_indomain.parquet")
        for path in (root / "base" / f"eval_{test}_indomain.jsonl",
                     root / f"{test}_indomain" / f"eval_{test}.jsonl"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in verdict_rows))

    gap_curves(root, pairs_dir, {"labels": ["indomain"]}, "Qwen3-8B", tmp_path)
    svg = tmp_path / "gap_curves_indomain_Qwen3-8B.svg"
    raw = svg.read_text()
    assert _svg_width_points(svg) <= 330
    assert "Cross-domain gap curves" not in raw
    assert "GPQA" in raw
    assert "Accuracy-logit gap" in raw
