import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import re
import xml.etree.ElementTree as ET

from capability_predictions_glm.judge_comparison import (
    build_tables,
    coefficient_alignment,
    coefficient_comparison,
    coefficient_statistics,
    plot_comparisons,
    strict_pair_accuracy,
)


def test_strict_pair_accuracy_excludes_seen_only_pairs() -> None:
    pairs = pd.DataFrame([
        {
            "bench": "b", "judge": "j", "model_left": "held", "model_right": "seen",
            "left_stronger": 1, "pick_left": 0, "d_signal": 1.0,
        },
        {
            "bench": "b", "judge": "j", "model_left": "seen", "model_right": "other",
            "left_stronger": 1, "pick_left": 1, "d_signal": -1.0,
        },
    ])
    coefficients = pd.DataFrame([
        {"benchmark": "b", "feature": "signal", "effective": 2.0},
    ])

    result = strict_pair_accuracy(pairs, coefficients, {"held"}).set_index("method")

    assert result.loc["pairwise_glm", "accuracy"] == 1.0
    assert result.loc["llm_judge", "accuracy"] == 0.0
    assert result.pair_count.eq(1).all()


def test_coefficient_comparison_matches_model_fold_and_averages_targets() -> None:
    capability = pd.DataFrame([
        {"fold": fold, "target_quartile": target, "benchmark": "b",
         "feature": "signal", "effective": value, "task": "pairwise"}
        for fold, values in (("model-fold-0", (1.0, 3.0)), ("model-fold-1", (5.0, 7.0)))
        for target, value in zip(("Q2", "Q4"), values)
    ])
    judges = pd.DataFrame([
        {"fold": "model-fold-0__item-0", "judge": "j", "benchmark": "b",
         "feature": "signal", "effective": 2.0},
        {"fold": "model-fold-1__item-0", "judge": "j", "benchmark": "b",
         "feature": "signal", "effective": 6.0},
    ])

    result = coefficient_comparison(capability, judges)

    assert result[["capability", "judge_coefficient"]].to_dict("records") == [
        {"capability": 2.0, "judge_coefficient": 2.0},
        {"capability": 6.0, "judge_coefficient": 6.0},
    ]
    assert result.model_fold.tolist() == ["model-fold-0", "model-fold-1"]


def test_cosine_contributions_keep_aligning_and_opposing_features() -> None:
    comparison = pd.DataFrame([
        {"fold": "model-fold-0__item-0", "model_fold": "model-fold-0",
         "item_fold": "item-0", "judge": "j", "benchmark": "b",
         "feature": "aligns", "capability": 1.0, "judge_coefficient": 1.0},
        {"fold": "model-fold-0__item-0", "model_fold": "model-fold-0",
         "item_fold": "item-0", "judge": "j", "benchmark": "b",
         "feature": "opposes", "capability": 1.0, "judge_coefficient": -1.0},
    ])

    cosines, contributions = coefficient_alignment(comparison)

    overall = cosines[cosines.benchmark == "all"].iloc[0]
    assert overall.cosine == 0.0
    values = contributions.set_index("feature").contribution
    np.testing.assert_allclose(values[["aligns", "opposes"]], [0.5, -0.5])
    assert np.isclose(contributions.contribution.sum(), overall.cosine)


def test_coefficient_statistics_report_correlation_and_all_quadrants() -> None:
    points = pd.DataFrame([
        {"judge": "j", "benchmark": "b", "feature": feature,
         "capability": capability, "judge_coefficient": judge, "observed": True}
        for feature, capability, judge in (
            ("positive_agreement", 1.0, 2.0),
            ("capability_positive", 1.0, -2.0),
            ("judge_positive", -1.0, 2.0),
            ("negative_agreement", -1.0, -2.0),
        )
    ])

    result = coefficient_statistics(points).set_index("benchmark")

    assert result.loc["all", "pearson"] == 0.0
    assert result.loc["all", "spearman"] == 0.0
    assert result.loc["all", "same_sign_share"] == 0.5
    assert result.loc["all", ["positive_positive", "positive_negative",
                              "negative_positive", "negative_negative"]].tolist() == [1, 1, 1, 1]


def test_comparison_figures_are_compact_and_keep_svg_text_editable(tmp_path) -> None:
    accuracy = pd.DataFrame([
        {"fold": f"model-fold-{fold}", "target_quartile": quartile, "bench": bench,
         "judge": "j", "method": method, "accuracy": value, "pair_count": 10}
        for fold in range(4)
        for quartile in ("Q2", "Q3", "Q4")
        for bench in ("aime", "gpqa", "intercode", "swe_bench_mini")
        for method, value in (("pairwise_glm", 0.8), ("llm_judge", 0.6))
    ])
    points = pd.DataFrame([
        {"judge": judge, "benchmark": bench, "feature": feature, "block": "surface",
         "capability": capability, "judge_coefficient": judge_value,
         "observed": not (judge == "k" and bench == "aime")}
        for judge in ("j", "k", "l", "m", "n", "o")
        for bench in ("aime", "gpqa", "intercode", "swe_bench_mini")
        for feature, capability, judge_value in (
            ("positive_agreement", 0.2, 0.3), ("capability_positive", 0.2, -0.3),
            ("judge_positive", -0.2, 0.3), ("negative_agreement", -0.2, -0.3),
        )
    ])
    cosine_judges = (
        "deepseek-r1-0528", "deepseek-v4-flash-0731", "gemma-4-31b-it",
        "gpt-oss-120b", "o4-mini",
    )
    cosines = pd.DataFrame([
        {"fold": f"model-fold-{fold}__item-{item}", "model_fold": f"model-fold-{fold}",
         "item_fold": f"item-{item}", "judge": judge, "benchmark": benchmark,
         "cosine": 0.4 if benchmark == "all" else 0.3,
         "observed": not (judge == "deepseek-r1-0528" and benchmark in ("all", "aime"))}
        for fold in range(4)
        for item in range(4)
        for judge in cosine_judges
        for benchmark in ("all", "aime", "gpqa", "intercode", "swe_bench_mini")
    ])
    contributions = pd.DataFrame([
        {"fold": f"model-fold-{fold}__item-{item}", "model_fold": f"model-fold-{fold}",
         "item_fold": f"item-{item}", "judge": "j", "feature": feature,
         "contribution": value}
        for fold in range(4)
        for item in range(4)
        for feature, value in (("aligns", 0.5), ("opposes", -0.1))
    ])
    eci = pd.DataFrame([
        {"judge": judge, "eci": value, "eci_ci_low": value - 1, "eci_ci_high": value + 1}
        for judge, value in zip(cosine_judges, (141.8, 152.9, 142.2, 140.5, 146.3))
    ])

    outputs = plot_comparisons(accuracy, points, cosines, contributions, eci, tmp_path)

    expected = {
        "accuracy-by-judge-overall", "accuracy-advantage-by-benchmark",
        "coefficient-alignment-summary", "coefficient-scatter-diagnostic",
        "cosine-vs-eci", "cosine-feature-drivers",
    }
    assert {path.stem for path in outputs} == expected
    for path in outputs:
        png = path.with_suffix(".png")
        svg = path.with_suffix(".svg")
        image = mpimg.imread(png)
        assert image.shape[1] <= 900
        assert image.shape[0] <= 840
        assert "<text" in svg.read_text()
        root = ET.fromstring(svg.read_text())
        font_sizes = [
            float(match.group(1))
            for element in root.iter()
            if element.tag.endswith("text")
            if (match := re.search(r"font-size: ([0-9.]+)px", element.attrib.get("style", "")))
        ]
        assert font_sizes and min(font_sizes) >= 7
        for group in root.iter():
            if group.attrib.get("id", "").startswith("PathCollection"):
                for element in group.iter():
                    style = element.attrib.get("style", "")
                    assert "stroke:" not in style or "stroke: none" in style
    accuracy_svg = (tmp_path / "accuracy-by-judge-overall.svg").read_text()
    assert ">Macro accuracy</text>" in accuracy_svg
    assert ">j</text>" in accuracy_svg
    assert ">0.5</text>" not in accuracy_svg
    summary_svg = (tmp_path / "coefficient-alignment-summary.svg").read_text()
    diagnostic_svg = (tmp_path / "coefficient-scatter-diagnostic.svg").read_text()
    advantage_svg = (tmp_path / "accuracy-advantage-by-benchmark.svg").read_text()
    assert "rotate(-90" in advantage_svg
    assert "rotate(-90" in summary_svg
    assert ">Pearson r</text>" in summary_svg
    assert ">Sign quadrants</text>" in summary_svg
    assert "same sign" in diagnostic_svg
    assert "s = same sign share" in diagnostic_svg
    assert "59% same sign" not in diagnostic_svg
    assert "fill-opacity: 0.14" in diagnostic_svg
    assert "positive agreement" not in diagnostic_svg
    diagnostic_root = ET.fromstring(diagnostic_svg)
    diagnostic_width = float(diagnostic_root.attrib["viewBox"].split()[2])
    right_ticks = []
    for element in diagnostic_root.iter():
        if not element.tag.endswith("text") or "x" not in element.attrib:
            continue
        try:
            value = float("".join(element.itertext()).replace("−", "-"))
        except ValueError:
            continue
        if value > 0:
            right_ticks.append(float(element.attrib["x"]))
    assert right_ticks and diagnostic_width - max(right_ticks) >= 8
    assert "Direct cells" not in (tmp_path / "cosine-vs-eci.svg").read_text()
    driver_image = mpimg.imread(tmp_path / "cosine-feature-drivers.png")
    assert driver_image.shape[0] >= 720
    assert driver_image.shape[1] <= 660


def test_build_tables_uses_q1_evidence_and_q2_to_q4_targets() -> None:
    responses = pd.DataFrame([
        {"bench": "b", "model": model, "item": item, "correct": correct,
         "is_canonical": True}
        for item, outcomes in (
            ("q1", {"held": 1, "seen": 1, "a": 1, "z": 1}),
            ("q2", {"held": 1, "seen": 0, "a": 1, "z": 1}),
            ("q3", {"held": 1, "seen": 0, "a": 0, "z": 0}),
            ("q4", {"held": 1, "seen": 0, "a": 0, "z": 0}),
        )
        for model, correct in outcomes.items()
    ])
    pairs = pd.DataFrame([
        {"bench": "b", "judge": "j", "item": "q1", "model_left": "held",
         "model_right": "seen", "pick_left": 0, "d_signal": 1.0},
    ])
    capability = pd.DataFrame([
        {"fold": "model-fold-0", "target_quartile": target, "benchmark": benchmark,
         "feature": "signal", "effective": 1.0, "task": "pairwise"}
        for target in ("Q2", "Q3", "Q4")
        for benchmark in ("b", "unsupported")
    ])
    judges = pd.DataFrame([
        {"fold": "model-fold-0__item-0", "judge": "j", "benchmark": "b",
         "feature": "signal", "effective": 1.0},
        {"fold": "model-fold-0__item-0", "judge": "j", "benchmark": "unsupported",
         "feature": "signal", "effective": 1.0},
    ])
    dictionary = pd.DataFrame([{"feature": "signal", "block": "surface"}])

    accuracy, points, cosines, contributions = build_tables(
        pairs, responses, capability, judges, dictionary,
        (("model-fold-0", ("held",)),),
    )

    assert set(accuracy.target_quartile) == {"Q2", "Q3", "Q4"}
    assert accuracy.set_index(["target_quartile", "method"]).accuracy.to_dict() == {
        (target, method): value
        for target in ("Q2", "Q3", "Q4")
        for method, value in (("pairwise_glm", 1.0), ("llm_judge", 0.0))
    }
    assert points[["benchmark", "feature", "block", "observed"]].to_dict("records") == [
        {"benchmark": "b", "feature": "signal", "block": "surface", "observed": True},
        {"benchmark": "unsupported", "feature": "signal", "block": "surface",
         "observed": False},
    ]
    assert set(cosines.benchmark) == {"all", "b", "unsupported"}
    assert not cosines.loc[cosines.benchmark == "unsupported", "observed"].item()
    assert contributions.feature.tolist() == ["signal"]
