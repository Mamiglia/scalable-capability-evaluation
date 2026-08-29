import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from . import fit
from .config import AnalysisConfig
from .fit_common import FitResult


def _matrix(tmp_path: Path) -> Path:
    root = tmp_path / "matrix"
    root.mkdir()
    pd.DataFrame([
        {"feature": "signal", "block": "surface"},
        {"feature": "noise", "block": "surface"},
    ]).to_parquet(
        root / "feature-dictionary.parquet", index=False
    )
    rows = []
    for item_index in range(4):
        for judge in ("j0", "j1"):
            signal = -1.0 if (item_index + (judge == "j1")) % 2 else 1.0
            for model_left in ("held", "seen"):
                rows.append({
                    "bench": "b0", "judge": judge, "item": f"i{item_index}",
                    "model_left": model_left, "model_right": "seen",
                    "split": "representation_train", "pick_left": int(signal > 0),
                    "d_signal": signal, "d_noise": 0.0,
                })
    pd.DataFrame(rows).to_parquet(root / "pair.parquet", index=False)
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(root)}))
    return config


def _set_processed_config(config: AnalysisConfig, matrix_config: Path) -> None:
    config.processed_matrix_dir.mkdir(parents=True)
    config.processed_matrix_config.write_text(matrix_config.read_text())


def test_fit_loads_the_processed_matrix_config_not_the_build_input(tmp_path: Path) -> None:
    """Fails if fitting falls back to the unprocessed design-matrix input."""
    processed_config = _matrix(tmp_path)
    config = AnalysisConfig(
        design_matrix_config=tmp_path / "missing-build-input.yaml",
        results_dir=tmp_path / "results", feature_blocks=("surface",),
        judges=("j0", "j1"), benchmarks=("b0",),
    ).validate()
    _set_processed_config(config, processed_config)

    rows, features = fit._load_rows(config)

    assert features == ["signal", "noise"]
    assert set(rows) >= {"signal", "noise", "pick_left"}


def test_frequentist_facade_import_does_not_require_hibayes() -> None:
    """Fails if CPU-only Slurm jobs eagerly import the Bayesian dependency stack."""
    code = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'scripts.judge_verdict_analysis.fit_hibayes' or name.startswith('hibayes'):
        raise ImportError('blocked Bayesian dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import scripts.judge_verdict_analysis.fit
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_local_synthetic_fit_writes_the_backend_fold_contract(tmp_path: Path) -> None:
    """Fails if orchestration stops writing a complete, non-overwriting fold."""
    config = AnalysisConfig(
        design_matrix_config=_matrix(tmp_path),
        results_dir=tmp_path / "results",
        feature_blocks=("surface",),
        judges=("j0", "j1"), benchmarks=("b0",),
        model_folds=(("held-models", ("held",)),), item_folds=2, inner_folds=2,
        elastic_net_cs=(1.0,), elastic_net_l1_ratios=(1.0,), ridge_cs=(1.0,),
        execution_mode="execute", execution_target="local_test", local_test_max_rows=32,
    ).validate()
    _set_processed_config(config, config.design_matrix_config)

    fit.run_fits(config)

    fold = config.fit_dir / "frequentist" / "held-models__item-0"
    assert sorted(path.name for path in fold.iterdir()) == [
        "coefficients.csv", "diagnostics.json", "manifest.json", "metrics.csv",
        "predictions.parquet", "selection.csv", "tuning.csv",
    ]
    manifest = json.loads((fold / "manifest.json").read_text())
    assert {key: manifest[key] for key in ("backend", "fold", "test_rows", "train_rows")} == {
        "backend": "frequentist", "fold": "held-models__item-0",
        "test_rows": 4, "train_rows": 4,
    }
    assert manifest["settings"]["elastic_net_cs"] == [1.0]
    assert manifest["settings"]["inner_folds"] == 2
    assert set(pd.read_parquet(fold / "predictions.parquet")) >= {"observed", "probability"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fit.run_fits(config)
    fit.run_fits(replace(config, resume_completed_folds=True))


def test_prepare_writes_remote_descriptors_without_loading_the_matrix(tmp_path: Path, monkeypatch) -> None:
    """Fails if default laptop preparation reads or copies full pair data."""
    config = AnalysisConfig(
        design_matrix_config=tmp_path / "not-read.yaml", results_dir=tmp_path / "results",
        model_folds=(("held-models", ("held",)),), item_folds=2,
        fit_backends=("frequentist",),
    ).validate()
    monkeypatch.setattr(fit, "_load_rows", lambda _: pytest.fail("prepare loaded pair data"))

    fit.run_fits(config)

    descriptor = config.fit_dir / "prepared"
    assert sorted(path.name for path in descriptor.iterdir()) == ["manifest.json"]
    manifest = json.loads((descriptor / "manifest.json").read_text())
    assert manifest["state"] == "prepared"
    assert manifest["scope"] == "full_plan"
    assert manifest["jobs"] == [
        {"backend": "frequentist", "fold": "held-models__item-0"},
        {"backend": "frequentist", "fold": "held-models__item-1"},
    ]
    assert manifest["target"] == "swc_hpc"
    assert manifest["source_paths"]["processed_matrix_config"] == str(config.processed_matrix_config)
    command = ast.parse(manifest["command"][-1])
    assert isinstance(command.body[0], ast.ImportFrom)
    assert isinstance(command.body[1].value, ast.Call)
    assert command.body[1].value.func.id == "run_remote_fits"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fit.run_fits(config)


def test_remote_entry_point_guards_before_loading_pair_rows(tmp_path: Path, monkeypatch) -> None:
    """Fails if the descriptor's remote entry point can bypass target validation."""
    config = AnalysisConfig(
        design_matrix_config=tmp_path / "not-read.yaml", results_dir=tmp_path / "results",
        execution_target="swc_hpc",
    ).validate()
    monkeypatch.setattr(fit, "_active_remote_target", lambda _: False)
    monkeypatch.setattr(fit, "_load_rows", lambda _: pytest.fail("remote guard loaded pair rows"))

    with pytest.raises(RuntimeError, match="refusing heavy local fit"):
        fit.run_remote_fits(config)


def test_swc_target_requires_a_slurm_compute_node(monkeypatch) -> None:
    """Fails if a Ceph-mounted gateway can satisfy the remote execution guard."""
    monkeypatch.setattr(fit, "_ceph_available", lambda: True)
    monkeypatch.setattr(fit.socket, "gethostname", lambda: "gpu-node-1")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert not fit._active_remote_target("swc_hpc")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert fit._active_remote_target("swc_hpc")
    monkeypatch.setattr(fit.socket, "gethostname", lambda: "sgw2")
    assert not fit._active_remote_target("swc_hpc")


def test_local_test_limit_refuses_before_loading_pair_rows(tmp_path: Path, monkeypatch) -> None:
    """Fails if an oversized local test reads the full pair matrix before refusing."""
    config = AnalysisConfig(
        design_matrix_config=_matrix(tmp_path), results_dir=tmp_path / "results",
        feature_blocks=("surface",), judges=("j0", "j1"), benchmarks=("b0",),
        model_folds=(("held-models", ("held",)),), item_folds=2,
        execution_mode="execute", execution_target="local_test", local_test_max_rows=1,
    ).validate()
    _set_processed_config(config, config.design_matrix_config)
    monkeypatch.setattr(fit, "_load_rows", lambda _: pytest.fail("oversized local test loaded pair rows"))

    with pytest.raises(RuntimeError, match="local_test_max_rows"):
        fit.run_fits(config)


def test_hibayes_job_uses_the_fold_local_frequentist_selection(tmp_path: Path, monkeypatch) -> None:
    """Fails if HiBayes is handed every processed feature instead of the screen."""
    config = AnalysisConfig(
        design_matrix_config=_matrix(tmp_path), results_dir=tmp_path / "results",
        feature_blocks=("surface",), judges=("j0", "j1"), benchmarks=("b0",),
        fit_backends=("hibayes",), hibayes_action="prepare",
        model_folds=(("held-models", ("held",)),), item_folds=2,
        execution_mode="execute", execution_target="local_test", local_test_max_rows=32,
    ).validate()
    _set_processed_config(config, config.design_matrix_config)
    monkeypatch.setattr(
        fit, "select_features", lambda *_: (["signal"], pd.DataFrame({"feature": ["signal"]})),
        raising=False,
    )

    fit.run_fits(config)

    job = yaml.safe_load((config.fit_dir / "hibayes" / "held-models__item-0" / "config.yaml").read_text())
    assert job["model"]["models"][0]["config"]["predictors"] == ["signal"]
    assert json.loads((config.fit_dir / "hibayes" / "held-models__item-0" / "manifest.json").read_text())["selected_features"] == ["signal"]


def test_hibayes_evaluation_adds_artifacts_to_an_existing_remote_job(tmp_path: Path, monkeypatch) -> None:
    """Fails if saved remote jobs cannot receive their evaluated artifacts."""
    config = AnalysisConfig(
        design_matrix_config=_matrix(tmp_path), results_dir=tmp_path / "results",
        feature_blocks=("surface",), judges=("j0", "j1"), benchmarks=("b0",),
        fit_backends=("hibayes",), hibayes_action="evaluate",
        model_folds=(("held-models", ("held",)),), item_folds=2,
        execution_mode="execute", execution_target="local_test", local_test_max_rows=32,
    ).validate()
    _set_processed_config(config, config.design_matrix_config)
    for item_fold in range(2):
        (config.fit_dir / "hibayes" / f"held-models__item-{item_fold}").mkdir(parents=True)

    def saved_result(job_dir: Path, test: pd.DataFrame, _: AnalysisConfig) -> FitResult:
        return FitResult(
            backend="hibayes", fold=job_dir.name, selected_features=("signal",),
            predictions=pd.DataFrame({"observed": test.pick_left, "probability": 0.5}),
            coefficients=pd.DataFrame(), tuning=pd.DataFrame(),
            metrics=pd.DataFrame({"metric": ["accuracy"], "value": [0.5]}), diagnostics={},
        )

    monkeypatch.setattr(fit, "evaluate_hibayes", saved_result)
    fit.run_fits(config)

    assert (config.fit_dir / "hibayes" / "held-models__item-0" / "manifest.json").exists()
