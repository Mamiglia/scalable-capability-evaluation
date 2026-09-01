import json
from pathlib import Path

import pandas as pd
import pytest

from capability_predictions_glm.sparse_comparison import (
    backproject_pc,
    build_variant_comparison,
    expand_composites,
)


def test_pc_backprojection_is_invariant_to_component_sign_and_order() -> None:
    loadings_a = pd.DataFrame([
        {"component": "pc_01", "feature": "c1", "loading": 1.0},
        {"component": "pc_01", "feature": "c2", "loading": 2.0},
        {"component": "pc_02", "feature": "c1", "loading": 3.0},
        {"component": "pc_02", "feature": "c2", "loading": 4.0},
    ])
    beta_a = pd.DataFrame({"feature": ["pc_01", "pc_02"], "effective": [5.0, 7.0]})
    loadings_b = pd.DataFrame([
        {"component": "other_1", "feature": "c1", "loading": -3.0},
        {"component": "other_1", "feature": "c2", "loading": -4.0},
        {"component": "other_2", "feature": "c1", "loading": 1.0},
        {"component": "other_2", "feature": "c2", "loading": 2.0},
    ])
    beta_b = pd.DataFrame({"feature": ["other_1", "other_2"], "effective": [-7.0, 5.0]})

    expected = pd.DataFrame({"feature": ["c1", "c2"], "effective": [26.0, 38.0]})
    pd.testing.assert_frame_equal(backproject_pc(beta_a, loadings_a), expected)
    pd.testing.assert_frame_equal(backproject_pc(beta_b, loadings_b), expected)

    with pytest.raises(ValueError, match="component inventory"):
        backproject_pc(beta_a.iloc[:1], loadings_a)
    with pytest.raises(ValueError, match="duplicate"):
        backproject_pc(beta_a, pd.concat([loadings_a, loadings_a.iloc[:1]]))


def test_composite_expansion_uses_benchmark_specific_signed_leaf_weights() -> None:
    provenance = {
        "source_blocks": {"a": "surface", "b": "rubric"},
        "final_composites": [{
            "output_id": "composite-001",
            "source_blocks": ["rubric", "surface"],
            "leaf_weights_by_benchmark": {
                "aime": {"a": 1.0},
                "gpqa": {"a": 0.5, "b": -0.5},
            },
        }],
    }
    coefficients = pd.DataFrame({"feature": ["composite-001"], "effective": [2.0]})

    assert expand_composites(coefficients, provenance, "aime").to_dict("records") == [
        {"feature": "a", "source_block": "surface", "effective": 2.0},
    ]
    assert expand_composites(coefficients, provenance, "gpqa").to_dict("records") == [
        {"feature": "a", "source_block": "surface", "effective": 1.0},
        {"feature": "b", "source_block": "rubric", "effective": -1.0},
    ]


def _write_comparison_fixture(
    root: Path, variant: str = "raw-l1",
) -> tuple[Path, Path, Path]:
    capability_root, judge_root, matrix_root = root / "capability", root / "judge", root / "matrix"
    aggregate = capability_root / "all" / variant / "aggregate"
    aggregate.mkdir(parents=True)
    rows = []
    for target, values in {
        "Q2": (1.0, 2.0), "Q3": (3.0, 2.0), "Q4": (2.0, 2.0),
    }.items():
        for feature, value in zip(("pc_a", "pc_b") if variant.startswith("pc-") else ("c1", "c2"), values):
            rows.append({
                "model_fold": "m0", "target_quartile": target, "benchmark": "x",
                "specification": "full", "kind": "reference", "feature": feature,
                "effective": value,
            })
        if variant.startswith("pc-"):
            cell = aggregate.parent / "cells" / f"m0__{target}"
            cell.mkdir(parents=True)
            pd.DataFrame([
                {"component": "pc_a", "feature": "c1", "loading": 1.0},
                {"component": "pc_a", "feature": "c2", "loading": 0.0},
                {"component": "pc_b", "feature": "c1", "loading": 0.0},
                {"component": "pc_b", "feature": "c2", "loading": 1.0},
            ]).to_csv(cell / "loadings.csv", index=False)
    pd.DataFrame(rows).to_csv(aggregate / "coefficients.csv", index=False)
    predictions = pd.DataFrame([
        {"model_fold": "m0", "target_quartile": target, "specification": "full",
         "kind": "reference", "bench": "x", "item": item,
         "model_left": left, "model_right": right, "observed": observed,
         "probability": probability}
        for target in ("Q2", "Q3", "Q4")
        for item, left, right, observed, probability in (
            ("i1", "held", "seen", 1, 0.8),
            ("i2", "seen", "held", 0, 0.7),
        )
    ])
    predictions.to_parquet(aggregate / "predictions.parquet", index=False)
    (aggregate / "settings.json").write_text(json.dumps({
        "settings": {
            "scope": "all", "feature_space": variant.split("-", 1)[0],
            "penalty": variant.rsplit("-", 1)[1],
        },
    }))

    judge_aggregate = judge_root / "fits" / variant / "aggregate"
    judge_aggregate.mkdir(parents=True)
    judge_coefficients = pd.DataFrame([
        {"fold": f"m0__item-{item_fold}", "feature": feature, "judge": "j1",
         "benchmark": "x", "effective": value}
        for item_fold in (0, 1)
        for feature, value in (
            (("other_1", -2.0), ("other_2", 2.0))
            if variant.startswith("pc-") else (("c1", 2.0), ("c2", 2.0))
        )
    ])
    judge_coefficients.to_csv(judge_aggregate / "coefficients.csv", index=False)
    if variant.startswith("pc-"):
        for item_fold in (0, 1):
            fold = judge_aggregate.parent / f"m0__item-{item_fold}"
            fold.mkdir()
            pd.DataFrame([
                {"component": "other_1", "feature": "c1", "loading": 0.0},
                {"component": "other_1", "feature": "c2", "loading": -1.0},
                {"component": "other_2", "feature": "c1", "loading": 1.0},
                {"component": "other_2", "feature": "c2", "loading": 0.0},
            ]).to_csv(fold / "loadings.csv", index=False)

    matrix_root.mkdir()
    (matrix_root / "composite-provenance.json").write_text(json.dumps({
        "source_blocks": {"a": "surface", "b": "rubric"},
        "final_composites": [
            {"output_id": "c1", "source_blocks": ["surface"],
             "leaf_weights_by_benchmark": {"x": {"a": 1.0}}},
            {"output_id": "c2", "source_blocks": ["rubric"],
             "leaf_weights_by_benchmark": {"x": {"b": 1.0}}},
        ],
    }))
    pd.DataFrame([
        {"bench": "x", "item": "i1", "model_left": "held", "model_right": "seen",
         "judge": "j1", "pick_left": 1, "delta_eci": 1.0},
        {"bench": "x", "item": "i2", "model_left": "held", "model_right": "seen",
         "judge": "j1", "pick_left": 1, "delta_eci": 3.0},
    ]).to_parquet(matrix_root / "pair.parquet", index=False)
    return capability_root, judge_root, matrix_root


def test_build_comparison_matches_model_fold_and_retains_item_fold_and_target(tmp_path: Path) -> None:
    tables = build_variant_comparison(*_write_comparison_fixture(tmp_path))

    assert set(tables) == {
        "coefficient_points", "coefficient_statistics", "fold_cosine",
        "leaf_cosine_contributions", "accuracy", "cosine_accuracy_eci",
    }
    points = tables["coefficient_points"]
    assert set(points.item_fold) == {0, 1}
    assert set(points.target_quartile) == {"Q2", "Q3", "Q4", "Q2-Q4 mean"}
    assert set(points.feature) == {"a", "b"}
    assert len(tables["fold_cosine"]) == 8
    contributions = tables["leaf_cosine_contributions"].groupby([
        "item_fold", "target_quartile",
    ], as_index=False).cosine_contribution.sum()
    checked = contributions.merge(
        tables["fold_cosine"][["item_fold", "target_quartile", "cosine"]],
        on=["item_fold", "target_quartile"], validate="one_to_one",
    )
    assert checked.cosine_contribution.tolist() == pytest.approx(checked.cosine.tolist())
    accuracy = tables["accuracy"]
    cells = accuracy[accuracy.benchmark == "x"]
    assert cells.capability_accuracy.eq(0.5).all()
    assert cells.judge_accuracy.eq(1.0).all()
    assert cells.mean_abs_delta_eci.eq(2.0).all()
    assert set(tables["cosine_accuracy_eci"].item_fold) == {0, 1}


def test_build_comparison_reconstructs_independent_pc_bases_before_matching(tmp_path: Path) -> None:
    tables = build_variant_comparison(*_write_comparison_fixture(tmp_path, "pc-ridge"))

    points = tables["coefficient_points"]
    q4 = points[(points.target_quartile == "Q4") & (points.item_fold == 0)]
    assert q4.set_index("feature")[[
        "capability_coefficient", "judge_coefficient",
    ]].to_dict("index") == {
        "a": {"capability_coefficient": 2.0, "judge_coefficient": 2.0},
        "b": {"capability_coefficient": 2.0, "judge_coefficient": 2.0},
    }
    assert tables["fold_cosine"].loc[
        tables["fold_cosine"].target_quartile == "Q4", "cosine",
    ].tolist() == pytest.approx([1.0, 1.0])
