import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from . import lesion_cells


def _predictions(probability: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "fold": ["f0"] * 4,
        "judge": ["j0", "j0", "j1", "j1"],
        "bench": ["b0"] * 4,
        "item": ["i0", "i1", "i0", "i1"],
        "observed": [1, 0, 1, 0],
        "probability": probability,
    })


def test_pair_predictions_uses_positive_helpful_lesion_and_singular_signs() -> None:
    full = _predictions([0.9, 0.1, 0.9, 0.1])
    intercept = _predictions([0.5] * 4)
    lesion = _predictions([0.6, 0.4, 0.6, 0.4])
    singular = _predictions([0.8, 0.2, 0.8, 0.2])

    lesion_rows = lesion_cells.pair_predictions(lesion, full, "full")
    singular_rows = lesion_cells.pair_predictions(singular, intercept, "intercept-only")

    assert lesion_rows.delta_log_loss.gt(0).all()
    assert singular_rows.delta_log_loss.gt(0).all()
    assert lesion_rows.delta_accuracy.eq(0).all()
    assert singular_rows.delta_accuracy.tolist() == [0.0, 1.0, 0.0, 1.0]
    np.testing.assert_allclose(
        lesion_rows.delta_log_loss, np.log(0.9) - np.log(0.6))
    np.testing.assert_allclose(
        singular_rows.delta_log_loss, np.log(0.8) - np.log(0.5))


def test_cell_summary_pools_judges_only_at_benchmark_scope() -> None:
    rows = pd.DataFrame({
        "fold": ["f0"] * 4,
        "judge": ["j0", "j0", "j1", "j1"],
        "bench": ["b0"] * 4,
        "item": ["i0", "i1", "i0", "i1"],
        "delta_log_loss": [1.0, 1.0, -1.0, -1.0],
        "delta_accuracy": [0.5, 0.5, -0.5, -0.5],
    })

    summary = lesion_cells.summarize_cells(rows, bootstrap_iterations=20, seed=3)

    benchmark = summary[summary.scope == "benchmark"].iloc[0]
    cells = summary[summary.scope == "judge×benchmark"].set_index("judge")
    assert benchmark.judge == ""
    assert benchmark.benchmark == "b0"
    assert benchmark.delta_log_loss == 0
    assert cells.loc["j0", "delta_log_loss"] == 1
    assert cells.loc["j1", "delta_log_loss"] == -1
    assert benchmark.fold_count == 1
    assert benchmark.row_count == 4


def test_cell_summary_bootstraps_items_with_all_repeated_rows() -> None:
    rows = pd.DataFrame({
        "fold": ["f0"] * 11,
        "judge": ["j0"] * 11,
        "bench": ["b0"] * 11,
        "item": ["repeated"] * 10 + ["single"],
        "delta_log_loss": [0.0] * 10 + [1.0],
        "delta_accuracy": [0.0] * 10 + [1.0],
    })

    summary = lesion_cells.summarize_cells(rows, bootstrap_iterations=1_000, seed=7)
    cell = summary[summary.scope == "judge×benchmark"].iloc[0]

    assert cell.delta_log_loss == 1 / 11
    assert cell.delta_log_loss_lower == 0
    assert cell.delta_log_loss_upper == 1
    assert cell.estimate == cell.delta_log_loss
    assert cell.lower == cell.delta_log_loss_lower
    assert cell.upper == cell.delta_log_loss_upper


def _artifact_index(tmp_path) -> tuple[Path, pd.DataFrame]:
    rows = []
    for fold in ("f0", "f1"):
        directory = tmp_path / fold
        directory.mkdir()
        paths = {}
        for name, probability in {
            "full": [0.9, 0.1, 0.9, 0.1],
            "intercept-only": [0.5] * 4,
            "lesion-a": [0.6, 0.4, 0.6, 0.4],
            "singular-b": [0.8, 0.2, 0.8, 0.2],
        }.items():
            path = directory / f"{name}.parquet"
            _predictions(probability).assign(fold=fold).to_parquet(path, index=False)
            paths[name] = path.relative_to(tmp_path)
        rows.extend([
            {"fold": fold, "specification": "full", "kind": "reference",
             "variable": "full", "reference": "", "artifact": "full",
             "predictions": paths["full"]},
            {"fold": fold, "specification": "intercept-only", "kind": "reference",
             "variable": "intercept-only", "reference": "", "artifact": "intercept-only",
             "predictions": paths["intercept-only"]},
            {"fold": fold, "specification": "lesion-a", "kind": "lesion-feature",
             "variable": "a", "reference": "full", "artifact": "lesion-a",
             "predictions": paths["lesion-a"]},
            {"fold": fold, "specification": "lesion-alias", "kind": "lesion-family",
             "variable": "alias", "reference": "full", "artifact": "lesion-a",
             "predictions": paths["lesion-a"]},
            {"fold": fold, "specification": "singular-b", "kind": "singular-feature",
             "variable": "b", "reference": "intercept-only", "artifact": "singular-b",
             "predictions": paths["singular-b"]},
        ])
    frame = pd.DataFrame(rows)
    path = tmp_path / "artifact-index.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_index_validation_resolves_references_and_aliases(tmp_path) -> None:
    path, _ = _artifact_index(tmp_path)

    plan, aliases = lesion_cells.validate_artifact_index(
        path, expected_folds=2, expected_labelled=3, expected_unique=2)

    assert aliases == {"lesion-alias": "lesion-a"}
    assert set(plan[plan.kind == "reference"].specification) == {
        "full", "intercept-only"}
    assert all(value.is_absolute() for value in plan.predictions)


def test_production_inventory_requires_exact_fold_kind_and_alias_counts() -> None:
    kinds = pd.Series(
        ["lesion-feature"] * 117 + ["singular-feature"] * 117
        + ["lesion-block"] * 9 + ["singular-block"] * 9
        + ["lesion-family"] * 5 + ["singular-family"] * 5)
    folds = [f"model-fold-{model}__item-{item}" for model in range(4) for item in range(4)]
    aliases = {
        "lesion-family-planning": "lesion-block-clause-plan",
        "singular-family-planning": "singular-block-clause-plan",
    }

    lesion_cells._validate_production_inventory(kinds, aliases, folds)
    with pytest.raises(ValueError, match="kind inventory"):
        lesion_cells._validate_production_inventory(kinds.iloc[:-1], aliases, folds)
    with pytest.raises(ValueError, match="planning aliases"):
        lesion_cells._validate_production_inventory(kinds, {}, folds)
    with pytest.raises(ValueError, match="fold inventory"):
        lesion_cells._validate_production_inventory(kinds, aliases, folds[:-1] + ["wrong"])


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing", "does not exist"),
        ("mixed", "plan differs"),
        ("wrong-reference", "must use full"),
        ("alias-path", "alias.*path"),
    ],
)
def test_index_validation_rejects_missing_corrupt_or_mixed_plans(
    tmp_path, corruption, message,
) -> None:
    path, frame = _artifact_index(tmp_path)
    if corruption == "missing":
        frame.loc[frame.specification == "full", "predictions"] = "missing.parquet"
    elif corruption == "mixed":
        frame.loc[(frame.fold == "f1") & (frame.specification == "lesion-a"), "variable"] = "other"
    elif corruption == "wrong-reference":
        frame.loc[frame.specification == "lesion-a", "reference"] = "intercept-only"
    else:
        frame.loc[(frame.fold == "f1") & (frame.specification == "lesion-alias"), "predictions"] = (
            frame.loc[(frame.fold == "f1") & (frame.specification == "singular-b"), "predictions"].iloc[0]
        )
    frame.to_csv(path, index=False)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        lesion_cells.validate_artifact_index(
            path, expected_folds=2, expected_labelled=3, expected_unique=2)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="judge"), "missing columns"),
        (lambda frame: frame.assign(observed=[2, 0, 1, 0]), "binary"),
        (lambda frame: frame.assign(probability=[np.nan, 0.1, 0.9, 0.1]), "finite"),
        (lambda frame: frame.assign(fold="wrong"), "fold"),
    ],
)
def test_prediction_validation_rejects_schema_and_semantic_corruption(
    tmp_path, mutate, message,
) -> None:
    path = tmp_path / "bad.parquet"
    mutate(_predictions([0.9, 0.1, 0.9, 0.1])).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=message):
        lesion_cells.read_predictions(path, "f0")


def test_pair_predictions_rejects_row_misalignment() -> None:
    reference = _predictions([0.9, 0.1, 0.9, 0.1])
    candidate = reference.iloc[[1, 0, 2, 3]].reset_index(drop=True)

    with pytest.raises(ValueError, match="row-aligned"):
        lesion_cells.pair_predictions(candidate, reference, "full")


def test_build_rejects_misaligned_full_and_intercept_references(tmp_path) -> None:
    index, frame = _artifact_index(tmp_path)
    intercept = tmp_path / frame.loc[
        (frame.fold == "f0") & (frame.specification == "intercept-only"), "predictions"
    ].iloc[0]
    singular = tmp_path / frame.loc[
        (frame.fold == "f0") & (frame.specification == "singular-b"), "predictions"
    ].iloc[0]
    for path in (intercept, singular):
        reordered = pd.read_parquet(path).iloc[[1, 0, 2, 3]].reset_index(drop=True)
        reordered.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="row-aligned"):
        lesion_cells.build(
            index, tmp_path / "cells", bootstrap_iterations=10,
            expected_inventory=(2, 3, 2),
        )


def test_build_writes_labelled_cell_summary_item_deltas_and_provenance(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = tmp_path / "cells"

    result = lesion_cells.build(
        index, output, bootstrap_iterations=50, seed=9, write_item_deltas=True,
        expected_inventory=(2, 3, 2),
    )

    assert result == output
    assert sorted(path.name for path in output.iterdir()) == [
        "cell-summary.csv", "input-signatures.csv", "item-cell-deltas.parquet",
        "manifest.json", "settings.json",
    ]
    summary = pd.read_csv(output / "cell-summary.csv", keep_default_na=False)
    assert set(summary.specification) == {"lesion-a", "lesion-alias", "singular-b"}
    assert set(summary.scope) == {"benchmark", "judge×benchmark"}
    assert {"kind", "variable", "reference", "judge", "benchmark",
            "fold_count", "row_count"}.issubset(summary)
    aliases = summary[summary.specification.isin(["lesion-a", "lesion-alias"])]
    pd.testing.assert_series_equal(
        aliases[aliases.specification == "lesion-a"].delta_log_loss.reset_index(drop=True),
        aliases[aliases.specification == "lesion-alias"].delta_log_loss.reset_index(drop=True),
        check_names=False,
    )
    item_deltas = pd.read_parquet(output / "item-cell-deltas.parquet")
    assert item_deltas.groupby(
        ["specification", "fold", "judge", "bench", "item"]
    ).size().eq(1).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["state"] == "completed"
    assert manifest["fold_count"] == 2
    assert manifest["labelled_specifications"] == 3
    assert manifest["unique_artifacts"] == 2
    assert manifest["source_artifact_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert manifest["artifact_sha256"]["cell-summary.csv"] == hashlib.sha256(
        (output / "cell-summary.csv").read_bytes()).hexdigest()


def test_completed_build_resumes_without_reopening_parquets(tmp_path, monkeypatch) -> None:
    index, _ = _artifact_index(tmp_path)
    output = tmp_path / "cells"
    kwargs = dict(
        bootstrap_iterations=10, seed=2, write_item_deltas=False,
        expected_inventory=(2, 3, 2),
    )
    lesion_cells.build(index, output, **kwargs)
    monkeypatch.setattr(
        lesion_cells, "read_predictions",
        lambda *_: (_ for _ in ()).throw(AssertionError("resume reopened a parquet")),
    )

    assert lesion_cells.build(index, output, **kwargs) == output
    assert not (output / "item-cell-deltas.parquet").exists()


def test_resume_rejects_changed_input_or_settings_and_never_overwrites(tmp_path) -> None:
    index, frame = _artifact_index(tmp_path)
    output = tmp_path / "cells"
    kwargs = dict(
        bootstrap_iterations=10, seed=2, write_item_deltas=False,
        expected_inventory=(2, 3, 2),
    )
    lesion_cells.build(index, output, **kwargs)
    summary_bytes = (output / "cell-summary.csv").read_bytes()

    with pytest.raises(RuntimeError, match="settings mismatch"):
        lesion_cells.build(index, output, **(kwargs | {"seed": 3}))
    changed = tmp_path / frame.loc[frame.specification == "full", "predictions"].iloc[0]
    saved = pd.read_parquet(changed)
    saved.assign(probability=[0.8, 0.2, 0.8, 0.2]).to_parquet(changed, index=False)
    with pytest.raises(RuntimeError, match="input provenance mismatch"):
        lesion_cells.build(index, output, **kwargs)
    assert (output / "cell-summary.csv").read_bytes() == summary_bytes


def test_resume_requires_manifest_hashes_for_every_expected_output(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = lesion_cells.build(
        index, tmp_path / "cells", bootstrap_iterations=10,
        expected_inventory=(2, 3, 2),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"].pop("cell-summary.csv")
    manifest_path.write_text(json.dumps(manifest))
    (output / "cell-summary.csv").unlink()

    with pytest.raises(RuntimeError, match="artifact inventory"):
        lesion_cells.build(
            index, output, bootstrap_iterations=10,
            expected_inventory=(2, 3, 2),
        )


def test_resume_rejects_corrupt_manifest_provenance(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = lesion_cells.build(
        index, tmp_path / "cells", bootstrap_iterations=10,
        expected_inventory=(2, 3, 2),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_artifact_index_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="completion manifest"):
        lesion_cells.build(
            index, output, bootstrap_iterations=10,
            expected_inventory=(2, 3, 2),
        )


def test_partial_directory_resumes_and_is_published_atomically(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = tmp_path / "cells"
    kwargs = dict(
        bootstrap_iterations=10, seed=2, write_item_deltas=False,
        expected_inventory=(2, 3, 2),
    )
    lesion_cells.build(index, output, **kwargs)
    stage = output.with_name(f".{output.name}.partial")
    output.rename(stage)
    (stage / "manifest.json").unlink()

    lesion_cells.build(index, output, **kwargs)

    assert output.is_dir()
    assert not stage.exists()
    assert json.loads((output / "manifest.json").read_text())["state"] == "completed"


def test_resume_rejects_parseable_but_corrupt_staged_summary(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = lesion_cells.build(
        index, tmp_path / "cells", bootstrap_iterations=10,
        expected_inventory=(2, 3, 2),
    )
    stage = output.with_name(f".{output.name}.partial")
    output.rename(stage)
    (stage / "manifest.json").unlink()
    summary = stage / "cell-summary.csv"
    summary.write_text("foreign\n")

    with pytest.raises(RuntimeError, match="summary schema"):
        lesion_cells.build(
            index, output, bootstrap_iterations=10,
            expected_inventory=(2, 3, 2),
        )
    assert summary.read_text() == "foreign\n"
    assert not output.exists()


def test_resume_refuses_an_unprovenanced_partial_without_overwriting_it(tmp_path) -> None:
    index, _ = _artifact_index(tmp_path)
    output = tmp_path / "cells"
    stage = output.with_name(f".{output.name}.partial")
    stage.mkdir()
    foreign = stage / "cell-summary.csv"
    foreign.write_text("foreign\n")

    with pytest.raises(RuntimeError, match="partial.*settings"):
        lesion_cells.build(
            index, output, bootstrap_iterations=10, seed=2,
            expected_inventory=(2, 3, 2),
        )
    assert foreign.read_text() == "foreign\n"
    assert not output.exists()


def test_cli_requires_explicit_paths_and_forwards_bootstrap_settings(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_build(index, output, **kwargs):
        captured.update(index=index, output=output, **kwargs)

    monkeypatch.setattr(lesion_cells, "build", fake_build)
    lesion_cells.main([
        "--artifact-index", str(tmp_path / "index.csv"),
        "--output-dir", str(tmp_path / "out"),
        "--bootstrap-iterations", "17", "--seed", "4", "--write-item-deltas",
    ])

    assert captured == {
        "index": tmp_path / "index.csv", "output": tmp_path / "out",
        "bootstrap_iterations": 17, "seed": 4, "write_item_deltas": True,
    }


def test_sbatch_refuses_execution_without_a_compute_allocation() -> None:
    script = Path(__file__).with_name("lesion_cells.sbatch")

    result = subprocess.run(
        ["bash", str(script), "index.csv", "output"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 2
    assert "Slurm allocation" in result.stderr


def test_sbatch_keeps_submit_relative_paths_after_entering_the_repo(tmp_path) -> None:
    script = Path(__file__).with_name("lesion_cells.sbatch")
    index = tmp_path / "index.csv"
    index.write_text("index\n")
    capture = tmp_path / "arguments.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    fake_python.chmod(0o755)
    env = os.environ | {
        "SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(tmp_path),
        "JV_CODE_DIR": str(Path(__file__).parents[2]), "JV_PYTHON": str(fake_python),
        "JV_PACKAGES": str(tmp_path), "CAPTURE": str(capture),
    }

    result = subprocess.run(
        ["bash", str(script), "index.csv", "cells"], cwd=tmp_path,
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--artifact-index") + 1] == str(index)
    assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path / "cells")
