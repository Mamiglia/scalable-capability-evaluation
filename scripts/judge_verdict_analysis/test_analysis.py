import hashlib
from pathlib import Path

import matplotlib
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")
from . import plots
from .associations import (
    block_correlations,
    capability_correlations,
    verdict_correlations,
)
from .basic_eda import (
    block_summary,
    correlation_pairs,
    duplicate_features,
    feature_summary,
)
from .build_dm import (
    build_design_matrix,
    build_expanded_design_matrix,
    write_design_matrix,
)
from .config import AnalysisConfig, CONFIG
from .data import list_blocks, load_features
from .feature_definitions import definitions_for
from .pca import align_pc_signs, pca_tables
from .plots import (
    _pair_label,
    _pca_loading_matrices,
    _verdict_matrices,
    plot_block_correlations,
    plot_capability_heatmap,
    plot_pca_loadings,
    plot_scree,
    plot_top_correlation_pairs,
    plot_verdict_heatmaps,
)


def _matrix(tmp_path: Path) -> Path:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    pd.DataFrame([
        {"feature": "length", "block": "surface", "source": "surface.parquet", "transform": "none", "n_missing": 0},
        {"feature": "hedging", "block": "surface", "source": "surface.parquet", "transform": "none", "n_missing": 0},
        {"feature": "rigour", "block": "rubric", "source": "rubric.parquet", "transform": "none", "n_missing": 1},
    ]).to_parquet(matrix / "feature-dictionary.parquet", index=False)
    pd.DataFrame({
        "bench": ["gpqa", "gpqa"], "model": ["a", "b"], "item": ["q", "q"],
        "split": ["representation_train", "confirmation"],
        "model_accuracy": [0.8, 0.6], "length": [2.0, 1.0],
        "hedging": [0.0, 1.0], "rigour": [4.0, 3.0],
    }).to_parquet(matrix / "transcript.parquet", index=False)
    pd.DataFrame({
        "bench": ["gpqa"], "judge": ["j"], "item": ["q"],
        "split": ["representation_train"], "pick_left": [1],
        "d_length": [1.0], "d_hedging": [-1.0], "d_rigour": [1.0],
    }).to_parquet(matrix / "pair.parquet", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(matrix)}))
    return config


def _matrix_with_scaling(tmp_path: Path) -> Path:
    config = _matrix(tmp_path)
    matrix = Path(yaml.safe_load(config.read_text())["output_root"])
    pd.DataFrame({
        "bench": ["a", "a", "b", "b"],
        "row_id": ["a1", "a2", "b1", "b2"],
        "split": ["representation_train"] * 4,
        "common": [0.0, 1.0, 1.0, 0.0],
        "duplicate": [0.0, 1.0, 1.0, 0.0],
        "domain_only": [0.0, 1.0, 0.0, 0.0],
    }).to_parquet(matrix / "transcript.parquet", index=False)
    pd.DataFrame({
        "bench": ["a"] * 4 + ["b"] * 4,
        "judge": ["j1", "j2"] * 4,
        "pair_id": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        "split": ["representation_train"] * 8,
        "pick_left": [1, 0] * 4,
        "d_common": [1.0, 1.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0],
        "d_duplicate": [1.0, 1.0, 2.0, 2.0, -1.0, -1.0, -2.0, -2.0],
        "d_domain_only": [1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    }).to_parquet(matrix / "pair.parquet", index=False)
    pd.DataFrame([
        {"feature": feature, "block": "surface", "source": "fixture", "transform": "none", "n_missing": 0}
        for feature in ("common", "duplicate", "domain_only")
    ]).to_parquet(matrix / "feature-dictionary.parquet", index=False)
    for name, prefix in (("scaling-transcript", ""), ("scaling-pair", "d_")):
        pd.DataFrame([
            {"benchmark": bench, "feature": f"{prefix}{feature}", "mean": 0.0, "sd": 1.0}
            for bench in ("a", "b") for feature in ("common", "duplicate", "domain_only")
        ]).to_parquet(matrix / f"{name}.parquet", index=False)
    return config


HISTORICAL_SURPRISAL = (
    "surprisal_local_mean__llama",
    "surprisal_full_bpc__llama",
    "surprisal_question_gain_bpc__llama",
    "surprisal_total_context_gain_bpc__llama",
    "surprisal_long_context_gain_bpc__llama",
    "surprisal_rolling_short_position_adjusted_mean__llama",
    "surprisal_rolling_long_position_adjusted_mean__llama",
    "surprisal_rolling_long_question_position_adjusted_mean__llama",
    "surprisal_rolling_context_gain_position_adjusted_mean__llama",
    "surprisal_rolling_question_gain_position_adjusted_mean__llama",
)
SHARED_SAE = tuple(f"sae_shared_{index}" for index in range(32))


def _expanded_matrix_fixture(tmp_path: Path) -> tuple[Path, Path]:
    baseline_root, source_root = tmp_path / "baseline", tmp_path / "source"
    baseline_root.mkdir()
    source_root.mkdir()
    baseline_features = ("readable_a", "readable_b")
    additions = HISTORICAL_SURPRISAL + SHARED_SAE
    blocks = {
        feature: ("embedding-sae-shared" if feature.startswith("sae_")
                  else "surprisal-position-adjusted" if "position_adjusted" in feature
                  else "surprisal-llama")
        for feature in additions
    }
    dictionary = pd.DataFrame([
        {"feature": feature, "block": "surface", "source": "readable", "transform": "none", "n_missing": 0}
        for feature in baseline_features
    ] + [
        {"feature": feature, "block": blocks[feature], "source": "canonical", "transform": "none", "n_missing": 0}
        for feature in additions
    ])
    dictionary.iloc[:2].assign(
        retained=True, original_block="surface", representative=baseline_features,
        processed_name=baseline_features,
    ).to_parquet(baseline_root / "feature-dictionary.parquet", index=False)
    dictionary.to_parquet(source_root / "feature-dictionary.parquet", index=False)

    transcript_metadata = pd.DataFrame({
        "bench": ["a", "b"], "model": ["m1", "m2"], "item": ["i1", "i2"],
        "split": ["representation_train"] * 2, "model_accuracy": [0.6, 0.7],
    })
    baseline_transcript = transcript_metadata.assign(readable_a=[1.0, 2.0], readable_b=[3.0, 4.0])
    source_transcript = baseline_transcript.assign(**{
        feature: [float(index), float(index + 1)] for index, feature in enumerate(additions)
    })
    baseline_transcript.to_parquet(baseline_root / "transcript.parquet", index=False)
    source_transcript.to_parquet(source_root / "transcript.parquet", index=False)

    pair_metadata = pd.DataFrame({
        "bench": ["a", "b"], "judge": ["j1", "j2"], "item": ["i1", "i2"],
        "model_left": ["m1", "m2"], "model_right": ["m2", "m1"],
        "split": ["representation_train"] * 2, "pick_left": [1, 0],
    })
    baseline_pair = pair_metadata.assign(d_readable_a=[1.0, -1.0], d_readable_b=[2.0, -2.0])
    source_pair = baseline_pair.assign(**{
        f"d_{feature}": [float(index), -float(index)] for index, feature in enumerate(additions)
    })
    baseline_pair.to_parquet(baseline_root / "pair.parquet", index=False)
    source_pair.to_parquet(source_root / "pair.parquet", index=False)

    for filename, prefix in (("scaling-transcript.parquet", ""), ("scaling-pair.parquet", "d_")):
        scaling = pd.DataFrame([
            {"benchmark": bench, "feature": f"{prefix}{feature}", "mean": 0.0, "sd": 1.0}
            for bench in ("a", "b") for feature in baseline_features + additions
        ])
        scaling[scaling.feature.isin([f"{prefix}{feature}" for feature in baseline_features])].to_parquet(
            baseline_root / filename, index=False)
        scaling.to_parquet(source_root / filename, index=False)

    pd.DataFrame({
        "feature": baseline_features, "status": ["retained"] * 2,
        "correlation_group": ["group-001", "group-002"],
        "representative": baseline_features, "retained": [True] * 2,
        "original_block": ["surface"] * 2, "group_size": [1] * 2,
        "processed_name": baseline_features,
    }).to_csv(baseline_root / "representative-map.csv", index=False)
    (baseline_root / "manifest.json").write_text(
        '{"source_hf_repo":"example/judging","source_hf_revision":"' + "a" * 40
        + '","source_status":"resolved"}\n')
    baseline_config = tmp_path / "baseline.yaml"
    source_config = tmp_path / "source.yaml"
    baseline_config.write_text(yaml.safe_dump({"output_root": str(baseline_root)}))
    source_config.write_text(yaml.safe_dump({"output_root": str(source_root)}))
    return baseline_config, source_config


@pytest.mark.parametrize("additions", [
    HISTORICAL_SURPRISAL,
    SHARED_SAE,
    HISTORICAL_SURPRISAL + SHARED_SAE,
])
def test_expanded_dm_preserves_baseline_and_appends_exact_features(
    tmp_path: Path, additions: tuple[str, ...],
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    baseline_root = Path(yaml.safe_load(baseline_config.read_text())["output_root"])
    output_root = tmp_path / "expanded"

    outputs = build_expanded_design_matrix(
        baseline_config, source_config, additions, output_root)

    inventory = ["readable_a", "readable_b", *additions]
    output_dictionary = outputs["feature-dictionary.parquet"]
    assert output_dictionary.feature.tolist() == inventory
    added_dictionary = output_dictionary.tail(len(additions))
    assert added_dictionary.retained.eq(True).all()
    assert added_dictionary.representative.tolist() == list(additions)
    assert added_dictionary.processed_name.tolist() == list(additions)
    assert added_dictionary.original_block.tolist() == added_dictionary.block.tolist()
    assert outputs["transcript.parquet"].columns.tolist() == [
        "bench", "model", "item", "split", "model_accuracy", *inventory]
    assert outputs["pair.parquet"].columns.tolist() == [
        "bench", "judge", "item", "model_left", "model_right", "split", "pick_left",
        *[f"d_{feature}" for feature in inventory],
    ]
    pd.testing.assert_frame_equal(
        outputs["transcript.parquet"].iloc[:, :7],
        pd.read_parquet(baseline_root / "transcript.parquet"),
    )
    pd.testing.assert_frame_equal(
        outputs["pair.parquet"].iloc[:, :9],
        pd.read_parquet(baseline_root / "pair.parquet"),
    )
    assert outputs["scaling-transcript.parquet"].feature.tolist() == [
        "readable_a", "readable_b", "readable_a", "readable_b",
        *additions, *additions,
    ]
    assert outputs["scaling-pair.parquet"].feature.str.removeprefix("d_").tolist() == \
        outputs["scaling-transcript.parquet"].feature.tolist()
    manifest = outputs["manifest.json"]
    assert manifest["features"] == {
        "baseline": 2, "added": len(additions), "retained": len(inventory),
        "inventory": inventory, "additions": list(additions),
    }
    assert manifest["source_hf_revision"] == "a" * 40
    assert manifest["source_sha256"]["canonical_config"] == hashlib.sha256(
        source_config.read_bytes()).hexdigest()
    assert manifest["source_sha256"]["baseline_manifest"] == hashlib.sha256(
        (baseline_root / "manifest.json").read_bytes()).hexdigest()
    assert manifest["source_sha256"]["baseline_representative_map"] == hashlib.sha256(
        (baseline_root / "representative-map.csv").read_bytes()).hexdigest()
    assert Path(yaml.safe_load(outputs["design-matrix.yaml"])["output_root"]) == output_root
    assert outputs["representative-map.csv"].tail(len(additions)).feature.tolist() == list(additions)
    assert outputs["representative-map.csv"].tail(len(additions)).status.eq("retained").all()

    write_design_matrix(outputs, output_root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_design_matrix(outputs, output_root)


@pytest.mark.parametrize(("artifact", "column", "message"), [
    ("feature-dictionary.parquet", "feature", "feature dictionary missing additions"),
    ("transcript.parquet", HISTORICAL_SURPRISAL[0], "transcript matrix missing additions"),
    ("pair.parquet", f"d_{HISTORICAL_SURPRISAL[0]}", "pair matrix missing additions"),
    ("scaling-transcript.parquet", HISTORICAL_SURPRISAL[0],
     "scaling-transcript.parquet missing additions"),
    ("scaling-pair.parquet", f"d_{HISTORICAL_SURPRISAL[0]}",
     "scaling-pair.parquet missing additions"),
])
def test_expanded_dm_rejects_additions_missing_from_any_source_artifact(
    tmp_path: Path, artifact: str, column: str, message: str,
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    source_root = Path(yaml.safe_load(source_config.read_text())["output_root"])
    table = pd.read_parquet(source_root / artifact)
    if artifact == "feature-dictionary.parquet":
        table = table[table.feature != HISTORICAL_SURPRISAL[0]]
    elif artifact.startswith("scaling-"):
        table = table[table.feature != column]
    else:
        table = table.drop(columns=column)
    table.to_parquet(source_root / artifact, index=False)

    with pytest.raises(ValueError, match=message):
        build_expanded_design_matrix(
            baseline_config, source_config, [HISTORICAL_SURPRISAL[0]], tmp_path / "expanded")


@pytest.mark.parametrize("level", ["transcript", "pair"])
def test_expanded_dm_rejects_source_row_or_metadata_mismatch(
    tmp_path: Path, level: str,
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    source_root = Path(yaml.safe_load(source_config.read_text())["output_root"])
    path = source_root / f"{level}.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "bench"] = "wrong"
    table.to_parquet(path, index=False)

    with pytest.raises(ValueError, match=f"source {level} rows/metadata do not match baseline"):
        build_expanded_design_matrix(
            baseline_config, source_config, [HISTORICAL_SURPRISAL[0]], tmp_path / "expanded")


@pytest.mark.parametrize(("artifact", "feature"), [
    ("scaling-transcript.parquet", HISTORICAL_SURPRISAL[0]),
    ("scaling-pair.parquet", f"d_{HISTORICAL_SURPRISAL[0]}"),
])
def test_expanded_dm_requires_each_addition_scaled_for_every_baseline_benchmark(
    tmp_path: Path, artifact: str, feature: str,
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    source_root = Path(yaml.safe_load(source_config.read_text())["output_root"])
    path = source_root / artifact
    scaling = pd.read_parquet(path)
    scaling = scaling[~((scaling.benchmark == "b") & (scaling.feature == feature))]
    scaling.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="additions lack baseline benchmark coverage"):
        build_expanded_design_matrix(
            baseline_config, source_config, [HISTORICAL_SURPRISAL[0]], tmp_path / "expanded")


@pytest.mark.parametrize(("additions", "message"), [
    ((HISTORICAL_SURPRISAL[0], HISTORICAL_SURPRISAL[0]), "additions must be unique"),
    (("readable_a",), "additions already present in baseline"),
])
def test_expanded_dm_rejects_duplicate_output_features(
    tmp_path: Path, additions: tuple[str, ...], message: str,
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_expanded_design_matrix(
            baseline_config, source_config, additions, tmp_path / "expanded")


@pytest.mark.parametrize(("artifact", "prefix"), [
    ("scaling-transcript.parquet", ""),
    ("scaling-pair.parquet", "d_"),
])
def test_expanded_dm_rejects_duplicate_scaling_rows(
    tmp_path: Path, artifact: str, prefix: str,
) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    source_root = Path(yaml.safe_load(source_config.read_text())["output_root"])
    path = source_root / artifact
    scaling = pd.read_parquet(path)
    row = scaling[
        (scaling.benchmark == "a")
        & (scaling.feature == f"{prefix}{HISTORICAL_SURPRISAL[0]}")
    ]
    pd.concat([scaling, row], ignore_index=True).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate benchmark/feature rows"):
        build_expanded_design_matrix(
            baseline_config, source_config, [HISTORICAL_SURPRISAL[0]], tmp_path / "expanded")


def test_expanded_dm_scaling_follows_requested_feature_order(tmp_path: Path) -> None:
    baseline_config, source_config = _expanded_matrix_fixture(tmp_path)
    additions = [HISTORICAL_SURPRISAL[1], HISTORICAL_SURPRISAL[0]]

    outputs = build_expanded_design_matrix(
        baseline_config, source_config, additions, tmp_path / "expanded")

    expected = additions + additions
    assert outputs["scaling-transcript.parquet"].tail(4).feature.tolist() == expected
    assert outputs["scaling-pair.parquet"].tail(4).feature.str.removeprefix("d_").tolist() == expected


def test_build_dm_writes_source_compatible_representative_matrix(tmp_path: Path) -> None:
    config = _matrix_with_scaling(tmp_path)
    settings = AnalysisConfig(
        design_matrix_config=config,
        results_dir=tmp_path / "results",
        feature_blocks=("surface",),
        benchmarks=("a", "b"),
        correlation_threshold=0.8,
        correlation_transitive=False,
        common_support=True,
    ).validate()
    outputs = build_design_matrix(settings)

    assert outputs["feature-dictionary.parquet"].feature.tolist() == ["common"]
    assert outputs["transcript.parquet"].bench.tolist() == ["a", "a", "b", "b"]
    assert outputs["representative-map.csv"].feature.tolist() == ["common", "duplicate", "domain_only"]
    assert outputs["representative-map.csv"].status.tolist() == [
        "retained", "grouped_out", "outside_common_support"]
    pd.testing.assert_frame_equal(
        pd.read_parquet(Path(yaml.safe_load(config.read_text())["output_root"]) / "transcript.parquet")
        [["bench", "row_id", "split"]],
        outputs["transcript.parquet"][["bench", "row_id", "split"]],
    )
    assert set(outputs) == {
        "transcript.parquet", "pair.parquet", "feature-dictionary.parquet",
        "scaling-transcript.parquet", "scaling-pair.parquet", "representative-map.csv",
        "manifest.json", "design-matrix.yaml",
    }

    write_design_matrix(outputs, settings.processed_matrix_dir)
    written = yaml.safe_load((settings.processed_matrix_dir / "design-matrix.yaml").read_text())
    assert Path(written["output_root"]) == settings.processed_matrix_dir


def test_loader_selects_blocks_and_maps_pair_columns(tmp_path: Path) -> None:
    config = _matrix(tmp_path)
    blocks = list_blocks(config)
    assert blocks.set_index("block").n_features.to_dict() == {"surface": 2, "rubric": 1}

    transcripts, dictionary = load_features("transcript", ["surface"], config=config)
    assert dictionary[["feature", "column"]].values.tolist() == [
        ["length", "length"], ["hedging", "hedging"]]
    assert {"bench", "model", "split", "model_accuracy", "length", "hedging"} <= set(transcripts)
    assert "rigour" not in transcripts

    pairs, dictionary = load_features("pair", ["rubric"], config=config)
    assert dictionary[["feature", "column"]].values.tolist() == [["rigour", "d_rigour"]]
    assert {"bench", "judge", "pick_left", "d_rigour"} <= set(pairs)


def test_loader_rejects_unknown_blocks(tmp_path: Path) -> None:
    config = _matrix(tmp_path)
    with pytest.raises(ValueError, match="unknown feature blocks: missing"):
        load_features("transcript", ["missing"], config=config)


def test_config_defaults_are_readable_and_validate_regressors(tmp_path: Path) -> None:
    assert "embedding-dense" not in CONFIG.feature_blocks
    assert "gpt-oss-120b" in CONFIG.judges
    assert CONFIG.min_reasoning_chars == 100
    assert CONFIG.glm_max_iter > 0
    assert CONFIG.pca_common_support
    assert CONFIG.pca_alignment == "stronger"
    assert CONFIG.correlation_threshold == 0.8
    assert not CONFIG.common_support
    assert CONFIG.fit_n_jobs == 2
    assert not CONFIG.resume_completed_folds
    assert CONFIG.processed_matrix_config == (
        Path(__file__).resolve().parents[2]
        / "judge_results/capability_predictions_glm/no-common-support/design-matrix/design-matrix.yaml")
    assert CONFIG.design_matrix_config.name == "design_matrices.yaml"
    assert CONFIG.data_dir == CONFIG.results_dir / "data"
    assert CONFIG.figure_dir == CONFIG.results_dir / "figures"
    assert CONFIG.basic_eda_dir == CONFIG.results_dir / "basic-eda"
    with pytest.raises(ValueError, match="glm_max_iter"):
        AnalysisConfig(glm_max_iter=0).validate()
    with pytest.raises(ValueError, match="correlation_threshold"):
        AnalysisConfig(correlation_threshold=1.1).validate()
    with pytest.raises(ValueError, match="fit_n_jobs"):
        AnalysisConfig(fit_n_jobs=0).validate()

    config = _matrix(tmp_path)
    table, dictionary = load_features(
        "transcript", ["surface"], features=["length"], config=config)
    assert dictionary.feature.tolist() == ["length"]
    assert "length" in table and "hedging" not in table


def test_feature_definitions_disambiguate_rubric_and_graph_depth() -> None:
    dictionary = pd.DataFrame({
        "feature": ["depth", "depth_max"],
        "block": ["rubric", "clause-motifs"],
    })
    got = definitions_for(dictionary).set_index("feature")
    assert "1–5" in got.loc["depth", "definition"]
    assert "longest" in got.loc["depth_max", "definition"].lower()
    assert got.loc["depth", "measurement"] == "LLM annotation"
    assert got.loc["depth_max", "measurement"] == "Deterministic"


def test_basic_eda_finds_constants_duplicates_and_correlations() -> None:
    table = pd.DataFrame({
        "bench": ["a"] * 4 + ["b"] * 4,
        "x": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        "same": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        "opposite": [-1.0, -2.0, -3.0, -4.0, -4.0, -3.0, -2.0, -1.0],
        "constant": [7.0] * 8,
        "constant2": [7.0] * 8,
        "partial": [1.0, None, 0.0, 1.0, 0.0, 1.0, None, 0.0],
    })
    dictionary = pd.DataFrame({
        "feature": ["x", "same", "opposite", "constant", "constant2", "partial"],
        "column": ["x", "same", "opposite", "constant", "constant2", "partial"],
        "block": ["surface", "surface", "surface", "rubric", "rubric", "rubric"],
    })

    features = feature_summary(table, dictionary)
    constant = features[(features.bench == "a") & (features.feature == "constant")].iloc[0]
    assert constant.n_unique == 1
    assert constant.is_constant
    assert features[(features.bench == "a") & (features.feature == "partial")].missing_rate.item() == 0.25

    blocks = block_summary(features)
    assert blocks[(blocks.bench == "a") & (blocks.block == "rubric")].n_constant.item() == 2

    duplicates = duplicate_features(table, dictionary)
    relations = {(r.feature_a, r.feature_b, r.relation) for r in duplicates.itertuples() if r.bench == "a"}
    assert ("x", "same", "identical") in relations
    assert ("x", "opposite", "sign_flipped") in relations
    assert not any("constant" in pair[:2] for pair in relations)

    correlations = correlation_pairs(table, dictionary, threshold=0.9, method="spearman")
    pairs = {(r.feature_a, r.feature_b, r.correlation) for r in correlations.itertuples() if r.bench == "a"}
    assert ("x", "same", 1.0) in pairs
    assert ("x", "opposite", -1.0) in pairs


def test_correlation_groups_are_shared_transitive_and_thresholded() -> None:
    from .preprocessing import correlation_groups

    pairs = pd.DataFrame([
        {"bench": bench, "feature_a": a, "feature_b": b,
         "abs_correlation": correlation}
        for bench, values in {
            "one": (("a", "b", 0.95), ("a", "c", 0.40), ("b", "c", 0.92)),
            "two": (("a", "b", 0.93), ("a", "c", 0.50), ("b", "c", 0.94)),
        }.items()
        for a, b, correlation in values
    ])
    dictionary = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "block": ["surface", "surface", "rubric", "rubric"],
    })

    grouped = correlation_groups(pairs, dictionary, threshold=0.9, transitive=True)
    groups = grouped.set_index("feature").correlation_group

    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
    assert grouped.set_index("feature").group_size.to_dict() == {
        "a": 3, "b": 3, "c": 3, "d": 1}


def test_correlation_groups_default_to_requiring_all_pairs_to_pass() -> None:
    from .preprocessing import correlation_groups

    pairs = pd.DataFrame([
        {"bench": bench, "feature_a": a, "feature_b": b,
         "abs_correlation": correlation}
        for bench, values in {
            "one": (("a", "b", 0.95), ("a", "c", 0.40), ("b", "c", 0.92)),
            "two": (("a", "b", 0.93), ("a", "c", 0.50), ("b", "c", 0.94)),
        }.items()
        for a, b, correlation in values
    ])
    dictionary = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "block": ["surface", "surface", "rubric"],
    })

    grouped = correlation_groups(pairs, dictionary, threshold=0.9)
    groups = grouped.set_index("feature").correlation_group

    assert groups["a"] == groups["b"]
    assert groups["c"] != groups["a"]


def test_common_support_requires_training_variance_in_every_benchmark() -> None:
    from .preprocessing import common_support

    table = pd.DataFrame({
        "bench": ["a"] * 3 + ["b"] * 3,
        "split": ["representation_train"] * 6,
        "common": [0, 1, 2, 2, 1, 0],
        "domain_only": [0, 1, 2, 0, 0, 0],
    })
    dictionary = pd.DataFrame({
        "feature": ["common", "domain_only"],
        "column": ["common", "domain_only"],
        "block": ["surface", "clause-plan"],
    })

    got = common_support(
        table, dictionary, ("a", "b"), ("representation_train",))

    assert got.feature.tolist() == ["common"]


def test_representatives_choose_shortest_name_and_honour_override() -> None:
    from .preprocessing import select_representatives

    groups = pd.DataFrame({
        "feature": ["clustering", "transitivity", "n_words"],
        "correlation_group": ["g1", "g1", "g2"],
        "group_size": [2, 2, 1],
        "original_block": ["clause-descriptors", "clause-descriptors", "surface"],
    })

    got = select_representatives(
        groups, "shortest_name", {"g1": "clustering"}, {})

    assert got.set_index("feature").retained.to_dict() == {
        "clustering": True, "transitivity": False, "n_words": True}


def test_associations_separate_judges_and_use_within_item_capability() -> None:
    pair = pd.DataFrame({
        "bench": ["b"] * 8,
        "judge": ["j1"] * 4 + ["j2"] * 4,
        "pick_left": [0, 0, 1, 1] * 2,
        "d_good": [0.0, 1.0, 2.0, 3.0] * 2,
        "d_bad": [3.0, 2.0, 1.0, 0.0] * 2,
    })
    pair_dictionary = pd.DataFrame({
        "feature": ["good", "bad"], "column": ["d_good", "d_bad"],
        "block": ["surface", "rubric"],
    })
    verdict = verdict_correlations(pair, pair_dictionary).set_index(["judge", "feature"])
    assert verdict.loc[("j1", "good"), "correlation"] > 0
    assert verdict.loc[("j2", "bad"), "correlation"] < 0

    rows = []
    for item in ("q1", "q2"):
        for model, score in (("weak", 0.2), ("mid", 0.5), ("strong", 0.8)):
            rows.append({"bench": "b", "item": item, "model": model,
                         "model_accuracy": score, "quality": score,
                         "reverse": -score})
    dictionary = pd.DataFrame({
        "feature": ["quality", "reverse"], "column": ["quality", "reverse"],
        "block": ["surface", "surface"],
    })
    capability = capability_correlations(pd.DataFrame(rows), dictionary, min_models=3)
    capability = capability.set_index("feature")
    assert capability.loc["quality", "within_item_rank_rho"] == pytest.approx(1.0)
    assert capability.loc["reverse", "within_item_rank_rho"] == pytest.approx(-1.0)


def test_pca_tables_and_block_correlations_are_readable() -> None:
    table = pd.DataFrame({
        "bench": ["a"] * 6,
        "f1": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "f2": [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0],
        "f3": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })
    dictionary = pd.DataFrame({
        "feature": ["f1", "f2", "f3"], "column": ["f1", "f2", "f3"],
        "block": ["surface", "surface", "rubric"],
    })
    variance, loadings = pca_tables(table, dictionary, n_components=2)
    assert variance.pc.tolist() == [1, 2]
    assert variance.cumulative.is_monotonic_increasing
    first = loadings[loadings.pc == 1].set_index("feature").abs_loading
    assert first["f1"] == pytest.approx(first["f2"])

    pairs = pd.DataFrame([
        {"bench": "a", "feature_a": "f1", "block_a": "surface",
         "feature_b": "f2", "block_b": "surface", "abs_correlation": 1.0},
        {"bench": "a", "feature_a": "f1", "block_a": "surface",
         "feature_b": "f3", "block_b": "rubric", "abs_correlation": 0.2},
    ])
    blocks = block_correlations(pairs).set_index(["bench", "block_a", "block_b"])
    assert blocks.loc[("a", "surface", "surface"), "mean_abs_correlation"] == 1.0
    assert blocks.loc[("a", "rubric", "surface"), "mean_abs_correlation"] == 0.2


def test_pca_common_support_excludes_domain_only_features_and_aligns_signs() -> None:
    table = pd.DataFrame({
        "bench": ["a"] * 4 + ["b"] * 4,
        "common": [0.0, 1.0, 2.0, 3.0] * 2,
        "domain_only": [0.0, 1.0, 0.0, 1.0] + [0.0] * 4,
    })
    dictionary = pd.DataFrame({
        "feature": ["common", "domain_only"],
        "column": ["common", "domain_only"],
        "block": ["surface", "clause-plan"],
    })
    _, loadings = pca_tables(table, dictionary, n_components=1, common_support=True)
    assert set(loadings.feature) == {"common"}

    associations = pd.DataFrame({
        "bench": ["a", "b"], "feature": ["common", "common"],
        "target_rho": [-0.5, 0.5],
    })
    aligned, directions = align_pc_signs(
        loadings, associations, "target_rho", alignment="stronger")
    direction = directions.set_index("bench").sign.to_dict()
    assert direction == {"a": -1, "b": 1}
    assert (directions.projection_after >= 0).all()
    assert set(aligned.alignment) == {"stronger"}


def test_nature_plots_write_png_and_editable_svg(tmp_path: Path) -> None:
    verdict = pd.DataFrame([
        {"bench": "a", "judge": judge, "feature": feature, "block": "surface",
         "correlation": value}
        for judge in ("j1", "j2")
        for feature, value in (("length", 0.3), ("errors", -0.4))
    ])
    capability = pd.DataFrame([
        {"bench": bench, "feature": feature, "block": "surface",
         "within_item_rank_rho": value}
        for bench in ("a", "b")
        for feature, value in (("length", 0.2), ("errors", -0.3))
    ])
    pairs = pd.DataFrame([
        {"bench": "a", "feature_a": "length", "block_a": "surface",
         "feature_b": "errors", "block_b": "rubric", "correlation": 1.0,
         "abs_correlation": 1.0},
    ])
    blocks = block_correlations(pairs)
    variance = pd.DataFrame([
        {"bench": bench, "pc": pc, "explained_variance_ratio": value,
         "cumulative": sum((0.6, 0.3)[:pc])}
        for bench in ("a", "b") for pc, value in ((1, 0.6), (2, 0.3))
    ])
    loadings = pd.DataFrame([
        {"bench": bench, "pc": pc, "feature": feature, "block": "surface",
         "loading": value, "abs_loading": abs(value)}
        for bench in ("a", "b") for pc in (1, 2)
        for feature, value in (("length", 0.7), ("errors", -0.6))
    ])

    plot_verdict_heatmaps(verdict, tmp_path, top=2)
    plot_capability_heatmap(capability, tmp_path, top=2)
    plot_block_correlations(blocks, tmp_path)
    plot_top_correlation_pairs(pairs, tmp_path, top=1)
    plot_scree(variance, tmp_path)
    plot_pca_loadings(loadings, tmp_path, pcs=2, top_per_pc=1)

    expected = ["verdict-correlations", "capability-correlations",
                "block-correlations", "top-feature-correlations", "pca-scree",
                "pca-loadings"]
    for name in expected:
        assert (tmp_path / f"{name}.png").exists()
        svg = (tmp_path / f"{name}.svg").read_text()
        assert "<text" in svg
    pair_svg = (tmp_path / "top-feature-correlations.svg").read_text()
    marker_group = pair_svg.split('<g id="PathCollection_1">', 1)[1].split(
        '<g id="line2d_', 1)[0]
    assert "clip-path" not in marker_group
    assert "Distinct-feature" in (tmp_path / "block-correlations.svg").read_text()


def test_correlation_group_plots_skip_singletons_and_show_benchmarks(tmp_path: Path) -> None:
    from .preprocessing import correlation_groups

    correlations = pd.DataFrame([
        {"bench": bench, "feature_a": a, "feature_b": b,
         "correlation": correlation, "abs_correlation": abs(correlation)}
        for bench, values in {
            "gpqa": (("a", "b", 0.95), ("a", "c", 0.40), ("b", "c", 0.92)),
            "intercode": (("a", "b", 0.93), ("a", "c", -0.50), ("b", "c", 0.94)),
            "swe_bench_mini": (("a", "b", 0.94), ("a", "c", 0.30), ("b", "c", 0.93)),
            "aime": (("a", "b", 0.92), ("a", "c", -0.20), ("b", "c", 0.95)),
        }.items()
        for a, b, correlation in values
    ])
    dictionary = pd.DataFrame({
        "feature": ["a", "b", "c", "d"],
        "block": ["surface", "surface", "rubric", "rubric"],
    })
    groups = correlation_groups(correlations, dictionary, threshold=0.9)

    plots.plot_correlation_groups(correlations, groups, tmp_path)

    output = tmp_path / "groupings"
    assert sorted(path.name for path in output.iterdir()) == ["group-001.png", "group-001.svg"]
    svg = (output / "group-001.svg").read_text()
    assert "<text" in svg
    assert svg.count(">a</text>") == 4
    assert ">SWE-bench</text>" in svg and ">Mini</text>" in svg
    assert ">Pearson r</text>" in svg


def test_pair_labels_are_single_line_and_bounded() -> None:
    label = _pair_label(
        "a_very_long_feature_name_that_would_wrap",
        "another_very_long_feature_name_that_would_wrap",
        "swe_bench_mini",
    )
    assert "\n" not in label
    assert len(label) <= 54


def test_verdict_panels_use_the_same_global_feature_rows() -> None:
    table = pd.DataFrame([
        {"bench": bench, "judge": "j", "feature": feature, "correlation": value}
        for bench, values in {
            "a": (("shared", 0.8), ("a_only", 0.7), ("b_only", 0.0)),
            "b": (("shared", 0.6), ("a_only", 0.0), ("b_only", 0.9)),
        }.items()
        for feature, value in values
    ])
    matrices = _verdict_matrices(table, top=2)
    assert [matrix.index.tolist() for _, matrix in matrices] == [
        ["shared", "b_only"], ["shared", "b_only"]]


def test_pca_panels_use_the_same_global_feature_rows() -> None:
    table = pd.DataFrame([
        {"bench": bench, "pc": pc, "feature": feature,
         "loading": value, "abs_loading": abs(value)}
        for bench in ("a", "b") for pc in (1, 2)
        for feature, value in (("shared", 0.8), ("small", 0.1), ("other", 0.6))
    ])
    matrices = _pca_loading_matrices(table, pcs=2, top=2)
    assert [matrix.index.tolist() for _, matrix in matrices] == [
        ["shared", "other"], ["shared", "other"]]
