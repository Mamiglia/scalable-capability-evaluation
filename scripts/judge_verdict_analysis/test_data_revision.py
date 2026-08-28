"""Regression tests for pinned Hugging Face source provenance."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from . import fit
from .build_dm import build_design_matrix
from .config import AnalysisConfig, CONFIG
from .test_analysis import _matrix_with_scaling


REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"


def _processed_config(tmp_path: Path, manifest: dict[str, object]) -> Path:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    config = tmp_path / "matrix.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(root)}))
    return config


def test_source_provenance_is_configured_and_propagated_to_matrix_and_plan(tmp_path: Path) -> None:
    """Fails if source pins are omitted, altered, or unresolved without a marker."""
    source_config = _matrix_with_scaling(tmp_path)
    settings = AnalysisConfig(
        design_matrix_config=source_config,
        results_dir=tmp_path / "results",
        feature_blocks=("surface",), benchmarks=("a", "b"),
        source_hf_repo="example/judging", source_hf_revision=REVISION,
    ).validate()

    tables = build_design_matrix(settings)
    matrix_yaml = yaml.safe_load(tables["design-matrix.yaml"])
    for record in (tables["manifest.json"], matrix_yaml):
        assert record["source_hf_repo"] == "example/judging"
        assert record["source_hf_revision"] == REVISION
        assert record["source_status"] == "resolved"

    unresolved = AnalysisConfig(
        design_matrix_config=tmp_path / "not-read.yaml", results_dir=tmp_path / "plan",
        source_hf_repo="example/judging", fit_backends=("frequentist",),
    ).validate()
    fit.run_fits(unresolved)
    plan = json.loads((unresolved.fit_dir / "prepared" / "manifest.json").read_text())
    assert plan["source_hf_repo"] == "example/judging"
    assert plan["source_hf_revision"] == ""
    assert plan["source_status"] == "unresolved/test-only"


def test_config_accepts_empty_revision_but_rejects_bad_source_identifiers() -> None:
    """Fails if an unresolved local fixture or malformed future pin is accepted wrongly."""
    assert CONFIG.source_hf_repo == "LASR-G5/judging"
    assert AnalysisConfig(source_hf_revision="").validate().source_hf_revision == ""
    with pytest.raises(ValueError, match="source_hf_repo"):
        AnalysisConfig(source_hf_repo="").validate()
    with pytest.raises(ValueError, match="source_hf_revision"):
        AnalysisConfig(source_hf_revision="main").validate()


@pytest.mark.parametrize(
    ("revision", "manifest", "error"),
    [
        ("", {}, "source_hf_revision"),
        (REVISION, {"source_hf_repo": "LASR-G5/judging", "source_hf_revision": "f" * 40}, "provenance"),
    ],
)
def test_remote_fit_rejects_unpinned_or_mismatched_source_before_loading(
    tmp_path: Path, monkeypatch, revision: str, manifest: dict[str, str], error: str,
) -> None:
    """Fails if a remote fit can load processed rows before provenance agrees."""
    config = AnalysisConfig(
        design_matrix_config=_processed_config(tmp_path, manifest), results_dir=tmp_path / "results",
        execution_mode="execute", execution_target="swc_hpc", source_hf_revision=revision,
    ).validate()
    monkeypatch.setattr(fit, "_active_remote_target", lambda _: True)
    monkeypatch.setattr(fit, "_load_rows", lambda _: pytest.fail("provenance guard loaded pair rows"))

    with pytest.raises(RuntimeError, match=error):
        fit.run_fits(config)


def test_local_test_allows_an_unresolved_synthetic_source(tmp_path: Path) -> None:
    """Fails if the local synthetic ceiling path is made to require a remote source pin."""
    root = tmp_path / "matrix"
    root.mkdir()
    pd.DataFrame({"value": [1]}).to_parquet(root / "pair.parquet", index=False)
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(yaml.safe_dump({"output_root": str(root)}))
    config = AnalysisConfig(
        design_matrix_config=config_path, results_dir=tmp_path / "results",
        execution_mode="execute", execution_target="local_test", local_test_max_rows=1,
    ).validate()

    fit._guard_execution(config)
