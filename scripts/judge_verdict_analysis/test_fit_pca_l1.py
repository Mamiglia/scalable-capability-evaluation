import importlib.util
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from .config import CONFIG


def test_pc_l1_module_exists() -> None:
    assert importlib.util.find_spec(
        "scripts.judge_verdict_analysis.fit_pca_l1"
    ) is not None


from . import fit_pca_l1  # noqa: E402  (existence check documents the first TDD boundary)


FEATURES = ["f0", "f1", "f2"]


def _rows() -> pd.DataFrame:
    rows = []
    for item_index in range(12):
        for judge_index, judge in enumerate(("j0", "j1")):
            sign = -1.0 if (item_index + judge_index) % 2 else 1.0
            for model_left in ("held", "seen"):
                rows.append({
                    "bench": "b0", "judge": judge, "item": f"i{item_index}",
                    "model_left": model_left, "model_right": "seen",
                    "fold": "held-models__item-0", "pick_left": int(sign > 0),
                    "f0": sign * 5, "f1": float((item_index % 3) - 1),
                    "f2": float((item_index % 4) - 2),
                })
    return pd.DataFrame(rows)


def _config(tmp_path=None, **changes):
    values = dict(
        results_dir=tmp_path or CONFIG.results_dir,
        judges=("j0", "j1"), benchmarks=("b0",),
        model_folds=(("held-models", ("held",)),), item_folds=2,
        inner_folds=2, elastic_net_cs=(0.001, 0.1), fit_n_jobs=1,
        glm_max_iter=1_000, fit_pca_components=2,
        fit_pca_algorithm="randomized", execution_mode="execute",
        execution_target="local_test", local_test_max_rows=200,
    )
    values.update(changes)
    return replace(CONFIG, **values)


def _train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _rows()
    return (
        rows[rows.model_left == "seen"].reset_index(drop=True),
        rows[rows.model_left == "held"].reset_index(drop=True),
    )


def test_outer_test_rows_cannot_change_fitted_components() -> None:
    train, test = _train_test()
    changed = test.copy()
    changed[FEATURES] = changed[FEATURES] * 1_000_000 + 12345

    _, _, loadings, variance = fit_pca_l1.fit_fold(
        train, test, FEATURES, 2, "randomized", _config())
    _, _, changed_loadings, changed_variance = fit_pca_l1.fit_fold(
        train, changed, FEATURES, 2, "randomized", _config())

    pd.testing.assert_frame_equal(loadings, changed_loadings)
    pd.testing.assert_frame_equal(variance, changed_variance)


def test_component_transform_is_odd() -> None:
    train, test = _train_test()
    model, _, _ = fit_pca_l1.fit_pooled_svd(
        train, FEATURES, 2, "randomized", random_seed=7)
    reversed_test = test.copy()
    reversed_test[FEATURES] *= -1

    transformed = fit_pca_l1.transform_components(test, FEATURES, model)
    reversed_transformed = fit_pca_l1.transform_components(
        reversed_test, FEATURES, model)

    np.testing.assert_array_equal(
        reversed_transformed.to_numpy(), -transformed.to_numpy())


def test_fold_fit_tunes_and_finishes_with_pure_l1() -> None:
    train, test = _train_test()

    result, selection, loadings, variance = fit_pca_l1.fit_fold(
        train, test, FEATURES, 2, "randomized", _config())

    assert result.backend == "pca-l1"
    assert set(result.tuning.stage) == {"l1"}
    assert set(result.tuning.l1_ratio) == {1.0}
    assert result.tuning.C.tolist() == [0.001, 0.1]
    assert result.tuning.selected.sum() == 1
    assert not any("ridge" in key or "hibayes" in key for key in result.diagnostics)
    assert result.diagnostics["l1_ratio"] == 1.0
    assert selection.feature.tolist() == ["pc_01", "pc_02"]
    assert selection.selected.sum() < len(selection)
    assert tuple(selection.loc[selection.selected, "feature"]) == result.selected_features
    assert set(result.coefficients.feature) == {"pc_01", "pc_02"}
    assert loadings.shape == (2 * len(FEATURES), 3)
    assert variance.shape == (2, 5)
    assert len(result.predictions) == len(test)


def test_l1_model_uses_saga_and_unit_l1_ratio() -> None:
    model = fit_pca_l1._l1_model(_config(), 0.1)

    assert model.solver == "saga"
    assert model.l1_ratio == 1.0


def test_tuning_and_final_convergence_are_recorded_separately() -> None:
    train, test = _train_test()

    result, _, _, _ = fit_pca_l1.fit_fold(
        train, test, FEATURES, 2, "randomized", _config(glm_max_iter=1))

    assert not result.tuning.converged.any()
    assert set(result.tuning.max_n_iter) == {1}
    assert result.diagnostics["l1_n_iter"] == 1
    assert result.diagnostics["l1_converged"] is False


def _patch_rows(monkeypatch, inventory) -> None:
    monkeypatch.setattr(fit_pca_l1, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit_pca_l1, "_load_rows", lambda _: tuple(inventory))


def _rehash(fold_dir: Path, *artifacts: str) -> None:
    manifest_path = fold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in artifacts:
        manifest["artifact_sha256"][artifact] = hashlib.sha256(
            (fold_dir / artifact).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_fold_checkpoints_and_aggregate_are_complete_and_resumable(
    tmp_path, monkeypatch,
) -> None:
    config = replace(_config(tmp_path), source_hf_revision="a" * 40)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    abandoned = (
        config.fit_dir / "pca-l1" / "folds" / ".held-models__item-0.partial")
    abandoned.mkdir(parents=True)
    (abandoned / "interrupted.txt").write_text("preempted\n")

    fold_dir = fit_pca_l1.run_fold(0, config)
    fit_pca_l1.run_fold(1, config)

    assert sorted(path.name for path in fold_dir.iterdir()) == [
        "coefficients.csv", "diagnostics.json", "manifest.json", "metrics.csv",
        "predictions.parquet", "selection.csv", "svd-explained-variance.csv",
        "svd-loadings.csv", "tuning.csv",
    ]
    manifest = json.loads((fold_dir / "manifest.json").read_text())
    assert manifest["backend"] == "pca-l1"
    assert manifest["settings"]["source_hf_revision"] == "a" * 40
    assert manifest["settings"]["processed_matrix_manifest_sha256"] == (
        config.processed_matrix_manifest_sha256)
    assert len(manifest["settings"]["source_rows_sha256"]) == 64
    assert len(manifest["settings"]["implementation_sha256"]) == 64
    assert manifest["settings"]["sklearn_version"]
    assert manifest["selected_features"] == pd.read_csv(
        fold_dir / "selection.csv").query("selected").feature.tolist()

    monkeypatch.setattr(
        fit_pca_l1, "fit_fold", lambda *_: (_ for _ in ()).throw(
            AssertionError("completed fold was recomputed")))
    assert fit_pca_l1.run_fold(0, config) == fold_dir

    output = fit_pca_l1.aggregate(config)
    assert len(pd.read_parquet(output / "predictions.parquet")) == 24
    assert json.loads((output / "manifest.json").read_text())["fold_count"] == 2


def test_resume_rejects_changed_source_or_feature_inventory(
    tmp_path, monkeypatch,
) -> None:
    config = replace(_config(tmp_path), source_hf_revision="a" * 40)
    rows = _rows().drop(columns="fold")
    inventory = [rows, FEATURES]
    _patch_rows(monkeypatch, inventory)
    fit_pca_l1.run_fold(0, config)

    with pytest.raises(RuntimeError, match="settings"):
        fit_pca_l1.run_fold(0, replace(config, source_hf_revision="b" * 40))

    inventory[0] = rows.assign(f3=rows.f1 * rows.f2)
    inventory[1] = [*FEATURES, "f3"]
    with pytest.raises(RuntimeError, match="settings"):
        fit_pca_l1.run_fold(0, config)


def test_resume_rejects_changed_values_or_labels_with_same_inventory(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    inventory = [rows, FEATURES]
    _patch_rows(monkeypatch, inventory)
    fit_pca_l1.run_fold(0, config)

    inventory[0] = rows.assign(f0=rows.f0 * 2, pick_left=1 - rows.pick_left)

    with pytest.raises(RuntimeError, match="settings"):
        fit_pca_l1.run_fold(0, config)


@pytest.mark.parametrize(("artifact", "mutate"), [
    ("predictions.parquet", lambda table: table.iloc[:-1]),
    ("tuning.csv", lambda table: table.assign(stage="ridge")),
    ("selection.csv", lambda table: table.assign(selected=True)),
])
def test_resume_rejects_semantically_corrupt_fold_artifacts(
    tmp_path, monkeypatch, artifact, mutate,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    path = fold_dir / artifact
    if path.suffix == ".parquet":
        mutate(pd.read_parquet(path)).to_parquet(path, index=False)
    else:
        mutate(pd.read_csv(path)).to_csv(path, index=False)
    _rehash(fold_dir, artifact)

    with pytest.raises(RuntimeError):
        fit_pca_l1.run_fold(0, config)


def test_resume_rejects_invalid_final_convergence_semantics(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    path = fold_dir / "diagnostics.json"
    diagnostics = json.loads(path.read_text())
    diagnostics["l1_converged"] = not diagnostics["l1_converged"]
    path.write_text(json.dumps(diagnostics))
    _rehash(fold_dir, path.name)

    with pytest.raises(RuntimeError):
        fit_pca_l1.run_fold(0, config)


@pytest.mark.parametrize(("artifact", "mutate"), [
    ("metrics.csv", lambda table: table.assign(value=0.5)),
    ("coefficients.csv", lambda table: table.assign(effective=999.0)),
    ("svd-loadings.csv", lambda table: table.assign(loading=0.0)),
])
def test_resume_rejects_self_consistent_hashes_with_wrong_semantics(
    tmp_path, monkeypatch, artifact, mutate,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    path = fold_dir / artifact
    mutate(pd.read_csv(path)).to_csv(path, index=False)
    _rehash(fold_dir, artifact)

    with pytest.raises(RuntimeError, match="artifact contents"):
        fit_pca_l1.run_fold(0, config)


def test_resume_rejects_predictions_from_the_wrong_fold(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    path = fold_dir / "predictions.parquet"
    predictions = pd.read_parquet(path).assign(fold="wrong-fold")
    predictions.to_parquet(path, index=False)
    _rehash(fold_dir, path.name)

    with pytest.raises(RuntimeError, match="artifact contents"):
        fit_pca_l1.run_fold(0, config)


def test_resume_rejects_incomplete_coefficient_cell_inventory(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    path = fold_dir / "coefficients.csv"
    coefficients = pd.read_csv(path).iloc[:-1]
    coefficients.to_csv(path, index=False)
    manifest_path = fold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coefficient_rows"] = len(coefficients)
    manifest_path.write_text(json.dumps(manifest))
    _rehash(fold_dir, path.name)

    with pytest.raises(RuntimeError):
        fit_pca_l1.run_fold(0, config)


def test_resume_rejects_nonminimal_selected_tuning_candidate(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fold_dir = fit_pca_l1.run_fold(0, config)
    tuning_path = fold_dir / "tuning.csv"
    tuning = pd.read_csv(tuning_path)
    tuning["mean_log_loss"] = [0.1, 0.9]
    wrong = 1
    tuning["selected"] = False
    tuning.loc[wrong, "selected"] = True
    tuning.to_csv(tuning_path, index=False)
    diagnostics_path = fold_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["l1_C"] = float(tuning.loc[wrong, "C"])
    diagnostics_path.write_text(json.dumps(diagnostics))
    _rehash(fold_dir, tuning_path.name, diagnostics_path.name)

    with pytest.raises(RuntimeError, match="artifact contents"):
        fit_pca_l1.run_fold(0, config)


def test_aggregate_rejects_changed_content_and_mixed_inventory(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    inventory = [rows, FEATURES]
    _patch_rows(monkeypatch, inventory)
    fit_pca_l1.run_fold(0, config)
    inventory[0] = rows.assign(f3=rows.f1 * rows.f2)
    inventory[1] = [*FEATURES, "f3"]
    fit_pca_l1.run_fold(1, config)

    with pytest.raises(RuntimeError, match="feature inventory"):
        fit_pca_l1.aggregate(config)


def test_aggregate_resume_rejects_content_different_from_shards(
    tmp_path, monkeypatch,
) -> None:
    config = _config(tmp_path)
    rows = _rows().drop(columns="fold")
    _patch_rows(monkeypatch, [rows, FEATURES])
    fit_pca_l1.run_fold(0, config)
    fit_pca_l1.run_fold(1, config)
    output = fit_pca_l1.aggregate(config)
    metrics = pd.read_csv(output / "metrics.csv")
    metrics.loc[0, "value"] += 0.25
    metrics.to_csv(output / "metrics.csv", index=False)

    with pytest.raises(RuntimeError, match="aggregate"):
        fit_pca_l1.aggregate(config)


def test_sbatch_has_valid_shell_and_requires_array_or_aggregate(tmp_path) -> None:
    script = Path(__file__).with_name("fit_pca_l1.sbatch")

    syntax = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    no_allocation = subprocess.run(
        ["bash", script, "fold", tmp_path, tmp_path],
        capture_output=True, text=True,
        env={key: value for key, value in os.environ.items()
             if not key.startswith("SLURM_")},
    )

    assert syntax.returncode == 0
    assert no_allocation.returncode == 2
    assert "Slurm allocation" in no_allocation.stderr
    source = script.read_text()
    assert "#SBATCH --array=0-15%4" in source
    assert "fit_pca_l1 aggregate" in source
    assert "hibayes" not in source.lower()
