import matplotlib.pyplot as plt
import pandas as pd

from capability_predictions_glm.contributions import _style as capability_style
from capability_predictions_glm.config import ROOT
from capability_predictions_glm.deterministic_views import (
    deterministic_feature_names,
    deterministic_group_rows,
    deterministic_rows,
    plot_deterministic_effect_scatter,
)
from scripts.judge_verdict_analysis.result_plots import _style as judge_style


def test_deterministic_subset_excludes_only_annotation_features_and_groups() -> None:
    dictionary = pd.DataFrame([
        {"feature": "length", "block": "surface"},
        {"feature": "bare_answer", "block": "structure"},
        {"feature": "graph", "block": "clause-descriptors"},
        {"feature": "rigour", "block": "rubric"},
        {"feature": "error", "block": "intermediate-errors"},
    ])
    names = deterministic_feature_names(dictionary)

    assert names == ("length", "graph")
    effects = pd.DataFrame({
        "variable": ["length", "bare_answer", "graph", "rigour", "error"],
    })
    assert deterministic_rows(effects, names, column="variable").variable.tolist() == [
        "length", "graph",
    ]
    groups = pd.DataFrame([
        {"variable_type": "block", "variable": "surface", "block": "surface"},
        {"variable_type": "block", "variable": "rubric", "block": "rubric"},
        {"variable_type": "family", "variable": "annotations", "block": "annotations"},
        {"variable_type": "family", "variable": "hand-engineered-text",
         "block": "hand-engineered-text"},
    ])
    assert deterministic_group_rows(groups).variable.tolist() == [
        "surface", "hand-engineered-text",
    ]


def test_real_active_deterministic_scope_has_95_features() -> None:
    dictionary = pd.read_parquet(
        ROOT / "judge_results/capability_predictions_glm/no-common-support-inputs/"
        "feature-dictionary.parquet"
    )
    names = deterministic_feature_names(dictionary)

    assert len(names) == 95
    assert "bare_answer" not in names
    assert not set(dictionary.loc[dictionary.feature.isin(names), "block"]) & {
        "rubric", "intermediate-errors",
    }


def test_dissection_styles_use_fill_only_markers() -> None:
    for style in (capability_style, judge_style):
        style()
        assert plt.rcParams["scatter.edgecolors"] == "none"
        assert plt.rcParams["lines.markeredgewidth"] == 0


def test_deterministic_effect_scatter_is_small_transparent_and_unlabelled(tmp_path) -> None:
    effects = pd.DataFrame([
        {"feature": feature, "singular": singular, "unique": unique}
        for feature, singular, unique in (
            ("one", 0.1, 0.2), ("two", -0.1, 0.1),
            ("three", 0.2, -0.1), ("four", -0.2, -0.2),
        )
    ])

    base = plot_deterministic_effect_scatter(effects, tmp_path)

    assert base.with_suffix(".png").exists()
    svg = base.with_suffix(".svg").read_text()
    assert "fill-opacity: 0.14" in svg
    assert all(f">{feature}</text>" not in svg for feature in effects.feature)
