from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import lesioning
from .config import CONFIG
from .fit_common import FitResult
from .fit_frequentist import fit_fixed_ridge
from .lesioning import (
    _bootstrap_summary,
    _checkpoint,
    _guard_execution,
    _paired_predictions,
    _prediction_rows,
    _resume_checkpoint,
    _validate_prediction_rows,
    _validate_specification_inventory,
    aggregate_lesions,
    build_specifications,
    lesion_fold,
    main,
    reference_specifications,
    run_fold,
    unique_specifications,
)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for bench in ("a", "b"):
        for item_index in range(12):
            signal = -2.0 if item_index % 2 == 0 else 2.0
            for judge in ("j1", "j2"):
                rows.append({
                    "bench": bench, "judge": judge, "item": f"item-{item_index}",
                    "fold": "synthetic", "pick_left": int(signal > 0),
                    "signal": signal, "noise": float((item_index + (bench == "b")) % 3 - 1),
                    "dropped": 0.0,
                })
    frame = pd.DataFrame(rows)
    return (
        frame[~frame.item.isin(("item-10", "item-11"))].reset_index(drop=True),
        frame[frame.item.isin(("item-10", "item-11"))].reset_index(drop=True),
    )


def _dictionary() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature": f"{block}-{index}", "block": block}
        for block, count in (
            ("surface", 13), ("structure", 18), ("rubric", 12),
            ("exposition", 8), ("derivation", 9), ("intermediate-errors", 9),
            ("clause-descriptors", 21), ("clause-motifs", 24), ("clause-plan", 3),
        )
        for index in range(count)
    ])


def test_fixed_selection_lesion_pairs_each_refit_with_the_full_predictions() -> None:
    """Fails if lesions retune, misalign rows, or leave derived feature terms behind."""
    train, test = _rows()
    full_predictions = test.drop(columns=["signal", "noise", "dropped", "pick_left"]).copy()
    full_predictions["observed"] = test.pick_left
    full_predictions["probability"] = np.where(test.pick_left == 1, 0.99, 0.01)
    full_fit = FitResult(
        backend="frequentist", fold="synthetic", selected_features=("signal", "noise", "dropped"),
        predictions=full_predictions, coefficients=pd.DataFrame(), tuning=pd.DataFrame(),
        metrics=pd.DataFrame(), diagnostics={"ridge_C": 1.0},
    )
    config = replace(CONFIG, glm_max_iter=2_000, random_seed=4)

    lesions = lesion_fold(full_fit, train, test, config)

    assert set(lesions.feature) == {"signal", "noise", "dropped"}
    assert lesions.groupby("feature").size().to_dict() == {
        "dropped": len(test), "noise": len(test), "signal": len(test),
    }
    assert lesions.groupby("feature").delta_log_loss.mean()["signal"] > lesions.groupby("feature").delta_log_loss.mean()["noise"]
    assert lesions.groupby("feature").delta_log_loss.mean()["signal"] > 0
    for _, lesion in lesions.groupby("feature"):
        pd.testing.assert_frame_equal(
            lesion.loc[:, full_predictions.columns].reset_index(drop=True),
            full_predictions.reset_index(drop=True),
        )


def test_fold_runner_tunes_once_reuses_c_and_resumes_completed_specs(tmp_path, monkeypatch) -> None:
    """Fails if a resumed fold retunes, refits aliases, or loses completed checkpoints."""
    train, test = _rows()
    train = train.drop(columns=["signal", "noise", "dropped"])
    test = test.drop(columns=["signal", "noise", "dropped"])
    dictionary = _dictionary()
    features = dictionary.feature.tolist()
    train = pd.concat([train, pd.DataFrame(0.0, index=train.index, columns=features)], axis=1)
    test = pd.concat([test, pd.DataFrame(0.0, index=test.index, columns=features)], axis=1)
    config = replace(
        CONFIG, results_dir=tmp_path / "results", execution_mode="execute", execution_target="local_test",
        local_test_max_rows=1_000, ridge_cs=(0.1,), bootstrap_iterations=2,
        model_folds=(("model-fold-0", ("held",)),), item_folds=1,
    )
    calls: list[tuple[str, ...]] = []
    ridge_calls: list[float] = []
    tuning_calls: list[tuple[str, ...]] = []

    def fixed_ridge(train_rows, test_rows, names, _, ridge_c):
        calls.append(tuple(names))
        ridge_calls.append(ridge_c)
        predictions = test_rows[["bench", "judge", "item", "fold"]].copy()
        predictions["observed"] = test_rows.pick_left
        predictions["probability"] = np.where(test_rows.pick_left == 1, 0.99, 0.01)
        coefficients = pd.DataFrame({
            "feature": list(names), "judge": "j1", "benchmark": "a",
            "population": 0.0, "judge_deviation": 0.0,
            "benchmark_deviation": 0.0, "cell_deviation": 0.0, "effective": 0.0,
        })
        return FitResult(
            backend="frequentist", fold="model-fold-0__item-0", selected_features=tuple(names),
            predictions=predictions, coefficients=coefficients, tuning=pd.DataFrame(),
            metrics=pd.DataFrame(), diagnostics={"ridge_C": ridge_c},
        )

    def tune(rows, names, *_):
        tuning_calls.append(tuple(names))
        return 0.1, pd.DataFrame({"C": [0.1], "selected": [True]})

    monkeypatch.setattr(lesioning, "_load_rows", lambda _: (pd.concat([train, test]), dictionary))
    monkeypatch.setattr(
        lesioning, "_select_outer_fold",
        lambda *_: ("model-fold-0__item-0", train, test),
    )
    monkeypatch.setattr(lesioning, "tune_ridge", tune)
    monkeypatch.setattr(lesioning, "fit_fixed_ridge", fixed_ridge)

    run_fold(0, config)
    run_fold(0, config)

    assert tuning_calls == [tuple(features)]
    assert calls[0] == tuple(features)
    assert calls[1] == ()
    assert len(calls) == 262  # full + intercept-only + 260 unique requested feature sets
    assert ridge_calls == [0.1] * 262
    fold_dir = config.lesion_dir / "folds/model-fold-0__item-0"
    specifications = pd.read_csv(fold_dir / "specifications.csv")
    assert len(specifications) == 262
    assert specifications.artifact.nunique() == 260
    assert len(list((fold_dir / "specifications").glob("*.parquet"))) == 260
    assert len(list((fold_dir / "coefficients").glob("*.parquet"))) == 260
    assert len(list((fold_dir / "item-metrics").glob("*.parquet"))) == 260
    assert len(list((fold_dir / "reference-coefficients").glob("*.parquet"))) == 2
    assert len(list((fold_dir / "reference-item-metrics").glob("*.parquet"))) == 2
    assert (fold_dir / "manifest.json").exists()
    with pytest.raises(RuntimeError, match="settings"):
        run_fold(0, replace(config, ridge_cs=(1.0,)))
    missing = fold_dir / "coefficients" / f"{specifications.artifact.iloc[0]}.parquet"
    missing.unlink()
    with pytest.raises(RuntimeError, match="missing completed checkpoint"):
        run_fold(0, config)


def test_clustered_summary_reports_both_deltas_and_the_configured_primary_metric() -> None:
    """Fails if bootstrap samples rows instead of benchmark/item clusters or drops accuracy."""
    rows = pd.DataFrame({
        "backend": ["frequentist"] * 4,
        "feature": ["signal"] * 4,
        "bench": ["a", "a", "b", "b"],
        "item": ["i0", "i0", "i1", "i1"],
        "fold": ["f0", "f1", "f0", "f1"],
        "delta_log_loss": [0.1, 0.5, 0.2, 0.2],
        "delta_accuracy": [0.0, 0.4, 0.2, 0.2],
    })
    summary = _bootstrap_summary(
        rows, replace(CONFIG, lesion_metric="accuracy", bootstrap_iterations=20, random_seed=7)
    ).iloc[0]

    assert summary.lesion_metric == "accuracy"
    assert summary.estimate == summary.delta_accuracy == 0.2
    assert summary.delta_log_loss == pytest.approx(0.25)
    assert summary.lower == summary.delta_accuracy_lower
    assert summary.upper == summary.delta_accuracy_upper
    assert summary.fold_count == 2
    assert summary.row_count == 4
    assert summary.delta_accuracy_lower == summary.delta_accuracy_upper == 0.2
    assert summary.delta_log_loss_lower <= summary.delta_log_loss <= summary.delta_log_loss_upper


def test_remote_lesion_guard_requires_a_non_gateway_slurm_job(monkeypatch) -> None:
    """Fails if a Ceph-mounted gateway can pass as an SWC compute allocation."""
    config = replace(CONFIG, execution_mode="execute", execution_target="swc_hpc")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="Slurm allocation"):
        _guard_execution(config)


def test_remote_lesion_guard_requires_the_configured_matrix_manifest_hash(
    tmp_path, monkeypatch,
) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "design-matrix.yaml").write_text(f"output_root: {matrix}\n")
    (matrix / "manifest.json").write_text('{"source":"wrong"}\n')
    config = replace(
        CONFIG, execution_mode="execute", execution_target="swc_hpc",
        processed_matrix_root=matrix, processed_matrix_manifest_sha256="0" * 64,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(lesioning.os, "uname", lambda: SimpleNamespace(nodename="compute-1"))

    with pytest.raises(RuntimeError, match="manifest hash"):
        _guard_execution(config)


def test_specifications_cover_every_requested_label_and_deduplicate_planning() -> None:
    dictionary = _dictionary()

    specifications = build_specifications(dictionary)
    unique, aliases = unique_specifications(specifications)
    references = reference_specifications(dictionary)

    assert len(specifications) == 262
    assert len(unique) == 260
    assert {spec.name for spec in references} == {"full", "intercept-only"}
    assert {spec.name for spec in specifications if spec.kind == "lesion-block"} == {
        f"lesion-block-{block}" for block in dictionary.block.unique()
    }
    assert {spec.name for spec in specifications if spec.kind == "singular-block"} == {
        f"singular-block-{block}" for block in dictionary.block.unique()
    }
    assert aliases == {
        "lesion-family-planning": "lesion-block-clause-plan",
        "singular-family-planning": "singular-block-clause-plan",
    }
    no_annotations = next(spec for spec in specifications if spec.name == "no-annotations")
    annotations_only = next(
        spec for spec in specifications if spec.name == "singular-family-annotations"
    )
    assert len(no_annotations.features) == 96
    assert len(annotations_only.features) == 21
    assert no_annotations.reference == "full"
    assert annotations_only.reference == "intercept-only"


def test_production_inventory_requires_exact_features_blocks_families_and_aliases() -> None:
    dictionary = _dictionary()
    specifications = build_specifications(dictionary)
    unique, aliases = unique_specifications(specifications)

    _validate_specification_inventory(dictionary, specifications, unique, aliases, CONFIG)
    with pytest.raises(ValueError, match="117 features"):
        _validate_specification_inventory(
            dictionary.iloc[:-1], specifications, unique, aliases, CONFIG)
    wrong_block = dictionary.copy()
    wrong_block.loc[0, "block"] = "wrong"
    with pytest.raises(ValueError, match="nine blocks"):
        _validate_specification_inventory(
            wrong_block, specifications, unique, aliases, CONFIG)
    wrong_families = replace(
        CONFIG,
        lesion_families=(("wrong", ("surface",)), *CONFIG.lesion_families[1:]),
    )
    with pytest.raises(ValueError, match="five families"):
        _validate_specification_inventory(
            dictionary, specifications, unique, aliases, wrong_families)


def test_specification_families_come_from_analysis_config() -> None:
    config = replace(CONFIG, lesion_families=(("surface-only", ("surface",)),))

    specifications = build_specifications(_dictionary(), config)
    families = [spec for spec in specifications if "family" in spec.kind]

    assert [(spec.kind, spec.variable) for spec in families] == [
        ("lesion-family", "surface-only"),
        ("singular-family", "surface-only"),
    ]


def test_bootstrap_primary_interval_matches_configured_metric() -> None:
    rows = pd.DataFrame({
        "backend": ["frequentist"] * 4,
        "specification": ["singular-feature-signal"] * 4,
        "kind": ["singular-feature"] * 4,
        "variable": ["signal"] * 4,
        "reference": ["intercept-only"] * 4,
        "bench": ["a", "a", "b", "b"],
        "item": ["i0", "i0", "i1", "i1"],
        "fold": ["f0", "f1", "f0", "f1"],
        "delta_log_loss": [0.1, 0.5, 0.2, 0.2],
        "delta_accuracy": [0.0, 0.4, 0.2, 0.2],
    })

    summary = _bootstrap_summary(
        rows, replace(CONFIG, lesion_metric="accuracy", bootstrap_iterations=20, random_seed=7)
    ).iloc[0]

    assert summary.specification == "singular-feature-signal"
    assert summary.kind == "singular-feature"
    assert summary.variable == "signal"
    assert summary.reference == "intercept-only"
    assert summary.estimate == summary.delta_accuracy == 0.2
    assert summary.lower == summary.delta_accuracy_lower
    assert summary.upper == summary.delta_accuracy_upper


def test_paired_predictions_use_full_for_lesions_and_null_for_singulars() -> None:
    metadata = pd.DataFrame({
        "bench": ["a", "a"], "item": ["i0", "i1"], "observed": [1, 0],
    })
    full = metadata.assign(probability=[0.9, 0.1])
    null = metadata.assign(probability=[0.5, 0.5])
    lesion = metadata.assign(probability=[0.6, 0.4])
    singular = metadata.assign(probability=[0.8, 0.2])
    lesion_spec = lesioning.Specification(
        "lesion-feature-x", "lesion-feature", "x", "full", ("y",)
    )
    singular_spec = lesioning.Specification(
        "singular-feature-x", "singular-feature", "x", "intercept-only", ("x",)
    )

    lesion_rows = _paired_predictions(lesion, full, lesion_spec, "fold-0")
    singular_rows = _paired_predictions(singular, null, singular_spec, "fold-0")

    assert lesion_rows.delta_log_loss.gt(0).all()
    assert singular_rows.delta_log_loss.gt(0).all()
    assert lesion_rows.reference.eq("full").all()
    assert singular_rows.reference.eq("intercept-only").all()
    assert lesion_rows.delta_accuracy.eq(0).all()
    assert singular_rows.delta_accuracy.tolist() == [0.0, 1.0]


def test_real_fixed_fits_normalize_reference_and_singular_prediction_columns() -> None:
    train, test = _rows()
    features = ["signal", "noise", "dropped"]
    config = replace(CONFIG, glm_max_iter=2_000, random_seed=4)
    intercept_only = fit_fixed_ridge(train, test, [], config, ridge_c=1.0)
    singular = fit_fixed_ridge(train, test, ["signal"], config, ridge_c=1.0)
    specification = lesioning.Specification(
        "singular-feature-signal", "singular-feature", "signal", "intercept-only", ("signal",)
    )

    reference_rows = _prediction_rows(intercept_only.predictions, features)
    singular_rows = _prediction_rows(singular.predictions, features)
    paired = _paired_predictions(singular_rows, reference_rows, specification, "synthetic")

    assert not set(features) & set(reference_rows)
    assert reference_rows.columns.tolist() == singular_rows.columns.tolist()
    assert len(paired) == len(test)


def test_aggregation_pairs_every_fold_with_the_declared_reference(tmp_path) -> None:
    config = replace(
        CONFIG, results_dir=tmp_path / "results", model_folds=(("model-fold-0", ("held",)),),
        item_folds=2, lesion_metric="accuracy", bootstrap_iterations=20,
    )
    specifications = pd.DataFrame([
        {"specification": "lesion-feature-x", "kind": "lesion-feature", "variable": "x",
         "reference": "full", "feature_count": 1, "artifact": "lesion-feature-x"},
        {"specification": "singular-feature-x", "kind": "singular-feature", "variable": "x",
         "reference": "intercept-only", "feature_count": 1, "artifact": "singular-feature-x"},
    ])
    for item_fold in range(2):
        fold = f"model-fold-0__item-{item_fold}"
        fold_dir = config.lesion_dir / "folds" / fold
        (fold_dir / "references").mkdir(parents=True)
        (fold_dir / "specifications").mkdir()
        (fold_dir / "reference-coefficients").mkdir()
        (fold_dir / "coefficients").mkdir()
        (fold_dir / "reference-item-metrics").mkdir()
        (fold_dir / "item-metrics").mkdir()
        specifications.to_csv(fold_dir / "specifications.csv", index=False)
        metadata = pd.DataFrame({
            "bench": ["a", "a"], "item": [f"i{item_fold}-0", f"i{item_fold}-1"],
            "observed": [1, 0],
        })
        predictions = {
            "references/full": metadata.assign(probability=[0.9, 0.1]),
            "references/intercept-only": metadata.assign(probability=[0.5, 0.5]),
            "specifications/lesion-feature-x": metadata.assign(probability=[0.6, 0.4]),
            "specifications/singular-feature-x": metadata.assign(probability=[0.8, 0.2]),
        }
        for name, frame in predictions.items():
            frame.to_parquet(fold_dir / f"{name}.parquet", index=False)
            prefix, artifact = name.split("/")
            coefficient_dir = "reference-coefficients" if prefix == "references" else "coefficients"
            item_dir = "reference-item-metrics" if prefix == "references" else "item-metrics"
            coefficient_features = [] if artifact == "intercept-only" else ["x"]
            pd.DataFrame({
                "feature": coefficient_features, "judge": "j1", "benchmark": "a",
                "population": 0.0, "judge_deviation": 0.0,
                "benchmark_deviation": 0.0, "cell_deviation": 0.0, "effective": 0.0,
            }).to_parquet(fold_dir / coefficient_dir / f"{artifact}.parquet", index=False)
            observed = frame.observed.to_numpy(int)
            probability = frame.probability.to_numpy(float)
            pd.DataFrame({
                "bench": frame.bench, "item": frame.item,
                "log_loss": -(observed * np.log(probability)
                              + (1 - observed) * np.log1p(-probability)),
                "accuracy": ((probability >= 0.5) == observed).astype(float),
                "row_count": 1,
            }).to_parquet(fold_dir / item_dir / f"{artifact}.parquet", index=False)
        (fold_dir / "manifest.json").write_text(
            f'{{"state":"completed","settings_signature":"fold-{item_fold}",'
            '"plan_signature":"shared-plan"}\n')

    output = aggregate_lesions(config)
    summary = pd.read_csv(output).set_index("specification")

    assert set(summary.index) == {"lesion-feature-x", "singular-feature-x"}
    assert summary.fold_count.eq(2).all()
    assert summary.delta_log_loss.gt(0).all()
    assert summary.loc["lesion-feature-x", "reference"] == "full"
    assert summary.loc["singular-feature-x", "reference"] == "intercept-only"
    assert (summary.lower == summary.delta_accuracy_lower).all()
    assert (summary.upper == summary.delta_accuracy_upper).all()
    assert (config.lesion_dir / "summary-manifest.json").exists()
    item_deltas = pd.read_parquet(config.lesion_dir / "item-deltas.parquet")
    artifact_index = pd.read_csv(config.lesion_dir / "artifact-index.csv")
    assert len(item_deltas) == 8
    assert {"specification", "kind", "variable", "reference", "delta_log_loss",
            "delta_accuracy", "row_count"} <= set(item_deltas)
    assert len(artifact_index) == 8  # two references + two labelled specs, across two folds
    assert artifact_index.coefficients.map(Path).map(Path.exists).all()
    assert artifact_index.item_metrics.map(Path).map(Path.exists).all()
    with pytest.raises(RuntimeError, match="summary settings"):
        aggregate_lesions(replace(config, lesion_metric="log_loss"))
    mixed = config.lesion_dir / "folds/model-fold-0__item-1/manifest.json"
    values = json.loads(mixed.read_text())
    values["plan_signature"] = "different-plan"
    mixed.write_text(json.dumps(values))
    with pytest.raises(RuntimeError, match="common plan"):
        aggregate_lesions(config)


def test_cli_scopes_fold_execution_and_summary_to_explicit_paths(tmp_path, monkeypatch) -> None:
    calls = []
    matrix = tmp_path / "matrix"
    results = tmp_path / "results"
    monkeypatch.setattr(
        lesioning, "run_fold",
        lambda fold_index, config: calls.append(("fold", fold_index, config)),
    )
    monkeypatch.setattr(
        lesioning, "aggregate_lesions",
        lambda config: calls.append(("summarize", None, config)),
    )

    main([
        "fold", "--fold-index", "3", "--processed-matrix-root", str(matrix),
        "--results-dir", str(results),
    ])
    main(["summarize", "--results-dir", str(results)])

    assert [(mode, index) for mode, index, _ in calls] == [("fold", 3), ("summarize", None)]
    fold_config = calls[0][2]
    assert fold_config.execution_mode == "execute"
    assert fold_config.execution_target == "swc_hpc"
    assert fold_config.processed_matrix_root == Path(matrix)
    assert fold_config.results_dir == Path(results)
    assert calls[1][2].results_dir == Path(results)


def test_resume_promotes_a_valid_deterministic_partial_without_rewriting(tmp_path) -> None:
    output = tmp_path / "candidate.parquet"
    partial = tmp_path / ".candidate.parquet.partial"
    expected = pd.DataFrame({"observed": [1], "probability": [0.75]})
    expected.to_parquet(partial, index=False)

    assert _resume_checkpoint(output, pd.read_parquet)
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)
    assert not partial.exists()


def test_checkpoint_lock_serializes_duplicate_writers_and_keeps_the_lock_file(tmp_path) -> None:
    output = tmp_path / "artifact.txt"
    entered = Event()
    release = Event()
    second_done = Event()
    writes = []

    def first_writer(path):
        writes.append("first")
        entered.set()
        assert release.wait(2)
        path.write_text("first")

    def second_writer(path):
        writes.append("second")
        path.write_text("second")

    def second_checkpoint():
        try:
            _checkpoint(output, second_writer, lambda path: path.read_text())
        finally:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _checkpoint, output, first_writer, lambda path: path.read_text())
        assert entered.wait(2)
        second = executor.submit(second_checkpoint)
        assert not second_done.wait(0.05)
        release.set()
        first.result()
        second.result()

    assert output.read_text() == "first"
    assert writes == ["first"]
    assert (tmp_path / ".artifact.txt.lock").exists()


def test_resumed_prediction_rows_must_match_the_held_out_schema() -> None:
    test = pd.DataFrame({
        "bench": ["a"], "item": ["i0"], "fold": ["f0"], "pick_left": [1],
        "signal": [0.5],
    })
    stale = pd.DataFrame({
        "bench": ["a"], "item": ["i0"], "fold": ["f0"], "observed": [1],
        "probability": [0.75], "signal": [0.5],
    })

    with pytest.raises(RuntimeError, match="prediction schema"):
        _validate_prediction_rows(stale, test, ["signal"])
