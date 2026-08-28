import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from . import fit
from .config import AnalysisConfig
from .fit_common import FitResult


def _matrix(tmp_path: Path) -> Path:
    root = tmp_path / "matrix"
    root.mkdir()
    pd.DataFrame([{"feature": "signal", "block": "surface"}]).to_parquet(
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
                    "d_signal": signal,
                })
    pd.DataFrame(rows).to_parquet(root / "pair.parquet", index=False)
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(root)}))
    return config


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


def test_hibayes_evaluation_adds_artifacts_to_an_existing_remote_job(tmp_path: Path, monkeypatch) -> None:
    """Fails if saved remote jobs cannot receive their evaluated artifacts."""
    config = AnalysisConfig(
        design_matrix_config=_matrix(tmp_path), results_dir=tmp_path / "results",
        feature_blocks=("surface",), judges=("j0", "j1"), benchmarks=("b0",),
        fit_backends=("hibayes",), hibayes_action="evaluate",
        model_folds=(("held-models", ("held",)),), item_folds=2,
        execution_mode="execute", execution_target="local_test", local_test_max_rows=32,
    ).validate()
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
