from dataclasses import replace
import json

import pandas as pd
import pytest

from scripts.judge_verdict_analysis.config import CONFIG

from . import pairwise_wrapper, pairwise_wrapper_lesions
from .pairwise_wrapper import fit_fold
from .pairwise_wrapper_lesions import (
    CapabilityLesionRun,
    capability_specifications,
    prepare_fold,
)
from .test_pairwise_wrapper import _config, _pairs, _responses, _sha256


def _prepared_run(monkeypatch, tmp_path, feature_space="raw", penalty="l1"):
    responses = tmp_path / "responses.parquet"
    _responses().to_parquet(responses, index=False)
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "rubric"],
    })
    config = _config(tmp_path, feature_space, penalty)
    provenance = {
        "source_paths": {"responses": str(responses)},
        "source_sha256": {"responses": _sha256(responses)},
    }
    run_arguments = {}
    if feature_space == "pc":
        loadings = config.pc_basis_root / "loadings.parquet"
        variance = config.pc_basis_root / "explained-variance.csv"
        provenance["source_paths"].update({
            "pc_basis": str(config.pc_basis_root),
            "pc_loadings": str(loadings),
            "pc_explained_variance": str(variance),
        })
        provenance["source_sha256"].update({
            "pc_loadings": _sha256(loadings),
            "pc_explained_variance": _sha256(variance),
        })
        run_arguments = {
            "pc_loadings_sha256": _sha256(loadings),
            "pc_explained_variance_sha256": _sha256(variance),
        }
    fit_fold(
        _pairs(), _responses(), dictionary, 0, config,
        manifest_provenance=provenance,
    )
    run = CapabilityLesionRun(
        config=config,
        responses=responses,
        responses_sha256=_sha256(responses),
        full_result_root=tmp_path,
        output_root=tmp_path / "lesions",
        **run_arguments,
    )
    monkeypatch.setattr(
        pairwise_wrapper_lesions, "_load_rows", lambda _: (_pairs(), dictionary),
    )
    monkeypatch.setattr(
        pairwise_wrapper, "tune_sparse",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retuned")),
    )
    return run, prepare_fold(0, run)


def test_capability_specifications_use_null_and_keep_raw_groups_pc_features_only() -> None:
    dictionary = pd.DataFrame({
        "feature": ["f0", "f1"],
        "block": ["mixed", "rubric"],
        "source_blocks": ['["surface", "rubric"]', '["rubric"]'],
    })
    raw = capability_specifications(
        dictionary,
        replace(CONFIG, feature_space="raw", lesion_families=(
            ("annotations", ("rubric",)),
        )),
    )
    pc = capability_specifications(
        pd.DataFrame({"feature": ["pc_001", "pc_002"], "block": "pc"}),
        replace(CONFIG, feature_space="pc"),
    )

    assert {spec.reference for spec in raw if spec.kind.startswith("singular-")} == {"null"}
    assert {spec.kind for spec in raw} == {
        "lesion-feature", "singular-feature", "lesion-block", "singular-block",
        "lesion-family", "singular-family",
    }
    assert {spec.kind for spec in pc} == {"lesion-feature", "singular-feature"}


@pytest.mark.parametrize("feature_space", ("raw", "pc"))
def test_prepare_fold_reconstructs_full_fit_and_uses_fixed_null_without_retuning(
    monkeypatch, tmp_path, feature_space,
) -> None:
    _, fold_dir = _prepared_run(monkeypatch, tmp_path, feature_space)

    null = pd.read_parquet(fold_dir / "predictions" / "null.parquet")
    specifications = pd.read_csv(
        fold_dir / "specifications.csv", keep_default_na=False,
    )
    frozen = json.loads((fold_dir / "frozen-hyperparameters.json").read_text())
    assert null.probability.eq(0.5).all()
    assert set(specifications.loc[
        specifications.kind.str.startswith("singular-"), "reference"
    ]) == {"null"}
    assert frozen == {"C": 1.0, "l1_ratio": 1.0}


@pytest.mark.parametrize("feature_space", ("raw", "pc"))
def test_specification_refits_use_capability_task_and_full_fit_hyperparameters(
    monkeypatch, tmp_path, feature_space,
) -> None:
    run, fold_dir = _prepared_run(monkeypatch, tmp_path, feature_space)
    specifications = pd.read_csv(fold_dir / "run-specifications.csv")

    for task_index in range(len(specifications)):
        pairwise_wrapper_lesions.run_specification(task_index, run)

    for specification in specifications.specification:
        diagnostics = json.loads(
            (fold_dir / "diagnostics" / f"{specification}.json").read_text()
        )
        assert diagnostics["C"] == 1.0
        assert diagnostics["l1_ratio"] == 1.0
        assert (fold_dir / "predictions" / f"{specification}.parquet").is_file()
        assert (fold_dir / "coefficients" / f"{specification}.parquet").is_file()
        assert (fold_dir / "item-metrics" / f"{specification}.parquet").is_file()

    pairwise_wrapper_lesions.finalize_fold(0, run)
    first = fold_dir / "predictions" / f"{specifications.specification.iloc[0]}.parquet"
    first.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="invalid completed fold"):
        pairwise_wrapper_lesions.finalize_fold(0, run)
