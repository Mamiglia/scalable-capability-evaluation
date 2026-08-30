from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from . import fit
from .build_dm import build_design_matrix, write_design_matrix
from .config import AnalysisConfig, CONFIG
from .data import load_features
from .fit_common import FitResult, prepare_features
from .fit_frequentist import JUDGE_TASK, fit_sparse
from .hierarchy import HierarchyLevels, hierarchical_design, outer_roles
from .lesioning import (
    Specification,
    _bootstrap_summary,
    _paired_predictions,
    build_specifications,
    prepare_lesions,
    reference_specifications,
)
from .preprocessing import apply_composites, fit_composites


def _dictionary(*features: str, blocks: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": features,
        "column": features,
        "block": blocks or ("surface",) * len(features),
    })


def _matrix_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    features = ("common", "duplicate", "domain_only")
    pd.DataFrame([
        {"feature": feature, "block": "rubric" if feature == "domain_only" else "surface", "source": "fixture",
         "transform": "none", "n_missing": 0}
        for feature in features
    ]).to_parquet(root / "feature-dictionary.parquet", index=False)
    transcript = pd.DataFrame({
        "bench": ["a", "a", "b", "b", "outside", "outside"],
        "row_id": ["a1", "a2", "b1", "b2", "o1", "o2"],
        "split": ["representation_train"] * 6,
        "common": [0, 1, 1, 0, 1, 0],
        "duplicate": [0, 1, 1, 0, 1, 0],
        "domain_only": [0, 1, 0, 0, 1, 0],
    })
    transcript.to_parquet(root / "transcript.parquet", index=False)
    pair = pd.DataFrame({
        "bench": np.repeat(["a", "b", "outside"], 4),
        "judge": ["j1", "j2"] * 6,
        "item": ["q0", "q0", "q1", "q1"] * 3,
        "pair_id": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2",
                    "o1", "o1", "o2", "o2"],
        "model_left": ["m1"] * 12,
        "model_right": ["m2"] * 12,
        "split": ["representation_train"] * 12,
        "pick_left": [1, 0] * 6,
        "d_common": [1, 1, 2, 2, -1, -1, -2, -2, -1, -1, -2, -2],
        "d_duplicate": [1, 1, 2, 2, -1, -1, -2, -2, -1, -1, -2, -2],
        "d_domain_only": [1, 1, 2, 2, 0, 0, 0, 0, -1, -1, -2, -2],
    })
    pair.to_parquet(root / "pair.parquet", index=False)
    for filename, prefix in (("scaling-transcript", ""), ("scaling-pair", "d_")):
        pd.DataFrame([
            {"benchmark": bench, "feature": f"{prefix}{feature}", "mean": 0.0, "sd": 1.0}
            for bench in ("a", "b", "outside") for feature in features
        ]).to_parquet(root / f"{filename}.parquet", index=False)
    config = tmp_path / "source.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(root)}))
    return config


def _fit_rows() -> tuple[pd.DataFrame, pd.DataFrame, list[str], AnalysisConfig]:
    rows = []
    for item_index in range(12):
        for judge_index, judge in enumerate(("j0", "j1")):
            sign = -1.0 if (item_index + judge_index) % 2 else 1.0
            for model_left in ("held", "seen"):
                rows.append({
                    "bench": "b0", "judge": judge, "item": f"i{item_index}",
                    "model_left": model_left, "model_right": "seen",
                    "fold": "held-models__item-0", "pick_left": int(sign > 0),
                    "f0": sign * 5, "f1": float(item_index % 3 - 1),
                    "f2": float(item_index % 4 - 2),
                })
    frame = pd.DataFrame(rows)
    config = replace(
        CONFIG, judges=("j0", "j1"), benchmarks=("b0",),
        model_folds=(("held-models", ("held",)),), item_folds=2,
        inner_folds=2, elastic_net_cs=(0.001, 0.1), fit_n_jobs=1,
        glm_max_iter=1_000, fit_pca_components=2,
        execution_mode="execute", execution_target="local_test", local_test_max_rows=200,
    )
    return (
        frame[frame.model_left == "seen"].reset_index(drop=True),
        frame[frame.model_left == "held"].reset_index(drop=True),
        ["f0", "f1", "f2"],
        config,
    )


def _pc_basis(tmp_path: Path, features: list[str]) -> Path:
    root = tmp_path / "pc-basis"
    root.mkdir()
    pd.DataFrame([
        {"component": component, "feature": feature, "loading": loading}
        for component, row in (
            ("pc_001", (1.0, 0.0, 0.0)),
            ("pc_002", (0.0, 0.6, 0.8)),
            ("pc_003", (0.0, -0.8, 0.6)),
        )
        for feature, loading in zip(features, row)
    ]).to_parquet(root / "loadings.parquet", index=False)
    pd.DataFrame({
        "component": ["pc_001", "pc_002", "pc_003"],
        "explained_variance": [3.0, 2.0, 1.0],
        "explained_variance_ratio": [0.5, 1 / 3, 1 / 6],
        "cumulative_explained_variance_ratio": [0.5, 5 / 6, 1.0],
        "singular_value": [3.0, 2.0, 1.0],
    }).to_csv(root / "explained-variance.csv", index=False)
    return root


def test_grouping_is_greedy_recomputed_hierarchical_and_replayable() -> None:
    e1 = np.array([1, -1, 0, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2, 0]) / np.sqrt(6)
    e3 = np.array([1, 1, 1, -3]) / np.sqrt(12)
    x = 0.079 / np.sqrt(0.19)
    table = pd.DataFrame({
        "bench": "x", "a": e1,
        "b": 0.9 * e1 + np.sqrt(0.19) * e2,
        "c": 0.79 * e1 + x * e2 + np.sqrt(1 - 0.79**2 - x**2) * e3,
    })
    dictionary = _dictionary("c", "a", "b", blocks=("rubric", "surface", "surface"))

    values, output_dictionary, provenance = fit_composites(table, dictionary)
    replayed = apply_composites(table, dictionary, provenance)
    _, _, shuffled = fit_composites(
        table.sample(frac=1, random_state=4), dictionary.sample(frac=1, random_state=5),
    )

    assert [(merge["left_parent_id"], merge["right_parent_id"]) for merge in provenance["merges"]] == [
        ("a", "b"), ("c", "composite-001"),
    ]
    assert provenance == shuffled
    assert provenance["merges"][-1]["leaf_weights"] == {"a": 0.25, "b": 0.25, "c": 0.5}
    assert output_dictionary[["feature", "block"]].values.tolist() == [["composite-002", "mixed"]]
    pd.testing.assert_frame_equal(values, replayed)


def test_grouping_handles_signs_benchmark_support_and_missing_rows() -> None:
    varying = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    table = pd.DataFrame({
        "bench": np.repeat(["both", "left", "right", "neither"], 5),
        "a": np.concatenate([[1, 2, np.nan, 4, 5], varying, np.zeros(10)]),
        "b": np.concatenate([[1, np.nan, 3, 4, 5], np.zeros(5), varying, np.zeros(5)]),
    })
    values, _, provenance = fit_composites(table, _dictionary("a", "b"))

    np.testing.assert_allclose(
        values["composite-001"],
        np.concatenate([[1, 2, 3, 4, 5], varying, varying, np.zeros(5)]),
    )
    assert provenance["merges"][0]["active_support"]["neither"]["output"] is False

    negative = pd.DataFrame({"bench": "x", "a": varying, "b": -varying})
    aligned, _, aligned_provenance = fit_composites(negative, _dictionary("a", "b"), threshold=1)
    np.testing.assert_allclose(aligned["composite-001"], varying)
    assert aligned_provenance["merges"][0]["branch_sign_flip"]
    disagreement = pd.concat([
        negative.assign(bench="x"), negative.assign(bench="y", b=varying),
    ], ignore_index=True)
    _, dictionary, rejected = fit_composites(disagreement, _dictionary("a", "b"))
    assert dictionary.feature.tolist() == ["a", "b"] and rejected["merges"] == []


def test_build_load_write_filter_scale_and_provenance_are_one_matrix_contract(tmp_path: Path) -> None:
    source_config = _matrix_fixture(tmp_path)
    config = AnalysisConfig(
        design_matrix_config=source_config, results_dir=tmp_path / "results",
        feature_blocks=("surface", "rubric"), benchmarks=("a", "b"),
        judges=("j1", "j2"),
    ).validate()

    outputs = build_design_matrix(config)
    provenance = outputs["composite-provenance.json"]
    assert set(outputs["pair.parquet"].bench) == {"a", "b"}
    assert outputs["feature-dictionary.parquet"][["feature", "block", "group_size"]].values.tolist() == [
        ["composite-002", "mixed", 3],
    ]
    assert outputs["transcript.parquet"].groupby("bench")["composite-002"].std().round(12).eq(1).all()
    digest = hashlib.sha256(json.dumps(
        provenance, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert outputs["manifest.json"]["composite_provenance_sha256"] == digest

    write_design_matrix(outputs, config.processed_matrix_dir)
    table, dictionary = load_features("pair", config=config.processed_matrix_config)
    assert dictionary.feature.tolist() == ["composite-002"]
    assert "d_composite-002" in table
    fit_rows, fit_dictionary = fit._load_rows(config)
    assert fit_dictionary[["feature", "block"]].values.tolist() == [["composite-002", "mixed"]]
    assert "composite-002" in fit_rows
    with pytest.raises(FileExistsError):
        write_design_matrix(outputs, config.processed_matrix_dir)


def test_feature_space_and_penalty_are_independent_config_axes(monkeypatch) -> None:
    for feature_space in ("raw", "pc"):
        for penalty in ("l1", "ridge"):
            config = replace(CONFIG, feature_space=feature_space, penalty=penalty).validate()
            assert (config.feature_space, config.penalty) == (feature_space, penalty)
    with pytest.raises(ValueError, match="feature_space"):
        replace(CONFIG, feature_space="ridge").validate()
    with pytest.raises(ValueError, match="penalty"):
        replace(CONFIG, penalty="elastic").validate()
    monkeypatch.setattr(fit, "_active_remote_target", lambda _: False)
    with pytest.raises(RuntimeError, match="refusing heavy local fit"):
        fit._guard_execution(replace(
            CONFIG, execution_mode="execute", execution_target="swc_hpc",
        ))


def test_outer_folds_and_hierarchy_keep_only_item_intercepts() -> None:
    roles = pd.DataFrame({
        "bench": ["a"] * 4,
        "item": ["i0", "i1", "i0", "i1"],
        "model_left": ["held", "seen", "held", "seen"],
        "model_right": ["seen", "seen", "seen", "held"],
    })
    assert outer_roles(roles, {"held"}, item_fold=0, item_folds=2).tolist() == [
        "test", "train", "test", "drop",
    ]
    frame = pd.DataFrame([
        {"judge": judge, "bench": bench, "item": item, "f": 1.0}
        for judge in ("j0", "j1") for bench in ("a", "b") for item in ("i0", "i1")
    ])
    penalties = {
        "judge_deviation_penalty": 4.0, "benchmark_deviation_penalty": 9.0,
        "cell_deviation_penalty": 16.0, "item_intercept_penalty": 25.0,
    }
    design, columns = hierarchical_design(
        frame, ("f",), HierarchyLevels.from_training(frame), penalties,
    )
    assert design.shape[0] == len(frame)
    assert columns.loc[columns.block.str.endswith("slope"), "item"].isna().all()


def test_pc_basis_is_loaded_once_and_shared_across_outer_folds(tmp_path: Path) -> None:
    train, test, features, config = _fit_rows()
    config = replace(config, pc_basis_root=_pc_basis(tmp_path, features))
    dictionary = pd.DataFrame({"feature": features, "block": "surface"})
    changed_train = train.copy()
    changed_train[features] = changed_train[features] * 1_000_000 + 12345

    prepared_train, prepared_test, prepared_dictionary, provenance = prepare_features(
        train, test, dictionary, "pc", config,
    )
    changed_prepared_train, changed_test, changed_dictionary, changed_provenance = prepare_features(
        changed_train, test, dictionary, "pc", config,
    )

    assert not prepared_train[["pc_001", "pc_002"]].equals(
        changed_prepared_train[["pc_001", "pc_002"]]
    )
    pd.testing.assert_frame_equal(prepared_test, changed_test)
    pd.testing.assert_frame_equal(prepared_dictionary, changed_dictionary)
    pd.testing.assert_frame_equal(provenance["loadings"], changed_provenance["loadings"])
    pd.testing.assert_frame_equal(
        provenance["explained_variance"], changed_provenance["explained_variance"],
    )
    assert prepared_dictionary.feature.tolist() == ["pc_001", "pc_002"]
    np.testing.assert_allclose(prepared_train.pc_001, train.f0)
    np.testing.assert_allclose(prepared_train.pc_002, 0.6 * train.f1 + 0.8 * train.f2)


def test_outer_fold_uses_shared_preparation_and_sparse_fit_for_both_axes(tmp_path: Path) -> None:
    assert JUDGE_TASK.outcome == "pick_left"
    assert JUDGE_TASK.group_columns == ("bench", "item")

    train, test, features, config = _fit_rows()
    config = replace(
        config, results_dir=tmp_path / "results",
        pc_basis_root=_pc_basis(tmp_path, features),
    )
    dictionary = pd.DataFrame({"feature": features, "block": "surface"})

    assert [field.name for field in fields(FitResult)] == [
        "backend", "fold", "selected_features", "predictions", "coefficients",
        "tuning", "metrics", "diagnostics",
    ]
    for feature_space, penalty in (("raw", "l1"), ("pc", "ridge")):
        variant = replace(config, feature_space=feature_space, penalty=penalty)
        result, prepared_dictionary, provenance = fit.fit_fold(
            train, test, dictionary, variant,
        )
        selected = result.tuning[result.tuning.selected].iloc[0]

        assert result.backend == f"{feature_space}-{penalty}"
        assert result.diagnostics["C"] == provenance["frozen_hyperparameters"]["C"] == selected.C
        if penalty == "l1":
            assert result.diagnostics["l1_ratio"] == provenance["frozen_hyperparameters"]["l1_ratio"] == selected.l1_ratio == 1
            assert (result.diagnostics["sklearn_penalty"], result.diagnostics["solver"]) == ("l1", "saga")
        else:
            assert result.diagnostics["l1_ratio"] is provenance["frozen_hyperparameters"]["l1_ratio"] is None
            assert pd.isna(selected.l1_ratio)
            assert (result.diagnostics["sklearn_penalty"], result.diagnostics["solver"]) == ("l2", "lbfgs")
        assert prepared_dictionary.feature.tolist() == (
            features if feature_space == "raw" else ["pc_001", "pc_002"]
        )
        fit._write_result(
            result, train, test, variant,
            prepared_dictionary=prepared_dictionary, provenance=provenance,
        )
        output = variant.fit_dir / result.backend / result.fold
        manifest = json.loads((output / "manifest.json").read_text())
        if feature_space == "pc":
            assert manifest["settings"]["pc_basis_root"] == str(variant.pc_basis_root)
        assert set(manifest["artifact_sha256"]) == {
            path.name for path in output.iterdir() if path.name != "manifest.json"
        }
        assert (output / "feature-provenance.json").is_file()
        assert (output / "feature-dictionary.parquet").is_file()
        assert fit._resume_existing(
            output, fit.ARTIFACTS,
            replace(variant, resume_completed_folds=True),
        )
    assert provenance["feature_space"] == "pc"


def test_feature_block_and_family_lesions_share_one_frozen_sparse_runner(tmp_path: Path) -> None:
    train, test, features, config = _fit_rows()
    dictionary = pd.DataFrame({
        "feature": features,
        "block": ["mixed", "surface", "rubric"],
        "source_blocks": [
            '["surface", "rubric"]', '["surface"]', '["rubric"]',
        ],
    })
    config = replace(
        config, feature_space="raw", penalty="l1",
        lesion_families=(("annotations", ("rubric",)),),
        pc_basis_root=_pc_basis(tmp_path, features),
    )

    specifications = build_specifications(dictionary, config)
    references = reference_specifications(dictionary)
    plan = prepare_lesions(train, test, dictionary, config)
    full = fit_sparse(
        plan["train"], plan["test"], list(references[0].features),
        config.penalty, plan["frozen_hyperparameters"], config, plan["tuning"],
    )
    lesion_spec = next(spec for spec in specifications if spec.name == "lesion-block-surface")
    lesion = fit_sparse(
        plan["train"], plan["test"], list(lesion_spec.features),
        config.penalty, plan["frozen_hyperparameters"], config,
    )

    assert {spec.kind for spec in specifications} == {
        "lesion-feature", "singular-feature", "lesion-block", "singular-block",
        "lesion-family", "singular-family",
    }
    assert {spec.name for spec in references} == {"full", "intercept-only"}
    assert lesion_spec.features == ("f2",)
    assert full.diagnostics["C"] == lesion.diagnostics["C"] == plan["frozen_hyperparameters"]["C"]
    assert full.diagnostics["l1_ratio"] == lesion.diagnostics["l1_ratio"] == 1.0
    assert plan["tuning"].selected.sum() == 1

    pc_plan = prepare_lesions(train, test, dictionary, replace(config, feature_space="pc"))
    assert {spec.kind for spec in pc_plan["specifications"]} == {
        "lesion-feature", "singular-feature",
    }


def test_lesion_deltas_and_item_clustered_uncertainty_share_reference_semantics() -> None:
    metadata = pd.DataFrame({
        "bench": ["a", "a"], "item": ["i0", "i1"], "observed": [1, 0],
    })
    specification = Specification("lesion-feature-x", "lesion-feature", "x", "full", ("y",))
    paired = _paired_predictions(
        metadata.assign(probability=[0.6, 0.4]),
        metadata.assign(probability=[0.9, 0.1]),
        specification, "f0",
    )
    rows = pd.concat([paired.assign(fold="f0"), paired.assign(fold="f1")], ignore_index=True)
    summary = _bootstrap_summary(
        rows, replace(CONFIG, bootstrap_iterations=20, random_seed=7),
    ).iloc[0]

    assert paired.delta_log_loss.gt(0).all()
    assert summary.reference == "full"
    assert summary.fold_count == 2 and summary.row_count == 4


def test_fit_aggregation_reports_item_and_both_fold_dimensions() -> None:
    predictions = pd.DataFrame({
        "fold": [f"m{model}__item-{item}" for model in range(2) for item in range(2) for _ in range(2)],
        "bench": ["a"] * 4 + ["b"] * 4,
        "judge": ["j0", "j1"] * 4,
        "item": ["q0", "q1"] * 4,
        "observed": [1, 0] * 4,
        "probability": [0.9, 0.2, 0.8, 0.4, 0.7, 0.3, 0.6, 0.1],
    })

    metrics, judge_benchmark, fold_metrics, uncertainty = fit.summarize_predictions(
        predictions, replace(CONFIG, bootstrap_iterations=20, random_seed=3),
    )

    assert set(metrics.metric) == {"accuracy", "log_loss"}
    assert set(judge_benchmark.columns) >= {"judge", "bench", "accuracy", "log_loss"}
    assert fold_metrics.fold.nunique() == 4
    assert set(uncertainty.method) == {
        "item_clustered", "model_fold", "item_fold", "conservative_fold",
    }
