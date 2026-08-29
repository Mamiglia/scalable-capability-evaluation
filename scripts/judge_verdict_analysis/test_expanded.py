import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from .build_dm import ARTIFACTS
from .config import CONFIG


def _expanded():
    return importlib.import_module("scripts.judge_verdict_analysis.expanded")


def test_build_materializes_the_three_exact_variants(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    calls = []
    monkeypatch.setattr(
        expanded, "build_expanded_design_matrix",
        lambda baseline, source, additions, output: calls.append(
            (baseline, source, tuple(additions), output)
        ) or {"variant": output.name},
    )
    monkeypatch.setattr(expanded, "write_design_matrix", lambda tables, output: None)
    baseline = tmp_path / "baseline.yaml"
    source = tmp_path / "source.yaml"
    output = tmp_path / "matrices"

    expanded.build_variants(baseline, source, output)

    assert [call[2] for call in calls] == [features for _, features in CONFIG.expanded_feature_sets]
    assert [len(call[2]) for call in calls] == [10, 32, 42]
    assert [call[3] for call in calls] == [output / name for name, _ in CONFIG.expanded_feature_sets]
    assert all(call[:2] == (baseline, source) for call in calls)


def _complete_variant(
    root: Path,
    name: str,
    additions: tuple[str, ...],
    baseline_config: Path | None = None,
    source_config: Path | None = None,
) -> Path:
    output = root / name
    output.mkdir(parents=True)
    inventory = ["readable", *additions]
    for artifact in ARTIFACTS:
        (output / artifact).write_text("fixture")
    manifest = {
        "settings": {
            "construction": "processed-baseline-plus-explicit-additions",
            "additions": list(additions),
            "correlation_grouping": False,
        },
        "features": {
            "baseline": 1,
            "added": len(additions),
            "retained": len(inventory),
            "inventory": inventory,
            "additions": list(additions),
        },
    }
    if baseline_config is not None and source_config is not None:
        manifest["source_paths"] = {
            "baseline_config": str(baseline_config),
            "canonical_config": str(source_config),
        }
        manifest["source_sha256"] = {
            key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for key, path in manifest["source_paths"].items()
        }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "design-matrix.yaml").write_text(yaml.safe_dump({"output_root": str(output)}))
    return output


def test_build_resumes_only_complete_matching_variants(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    output = tmp_path / "matrices"
    baseline, source = tmp_path / "baseline.yaml", tmp_path / "source.yaml"
    baseline.write_text("baseline")
    source.write_text("source")
    for name, additions in CONFIG.expanded_feature_sets:
        _complete_variant(output, name, additions, baseline, source)
    monkeypatch.setattr(
        expanded, "build_expanded_design_matrix",
        lambda *_: pytest.fail("complete variant was rebuilt"),
    )

    expanded.build_variants(baseline, source, output)


def test_build_refuses_complete_variants_from_stale_inputs(tmp_path: Path) -> None:
    expanded = _expanded()
    output = tmp_path / "matrices"
    baseline, source = tmp_path / "baseline.yaml", tmp_path / "source.yaml"
    baseline.write_text("baseline")
    source.write_text("source")
    for name, additions in CONFIG.expanded_feature_sets:
        _complete_variant(output, name, additions, baseline, source)
    baseline.write_text("changed baseline")

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded.build_variants(baseline, source, output)


def test_build_refuses_a_partial_existing_variant(tmp_path: Path) -> None:
    expanded = _expanded()
    output = tmp_path / "matrices"
    name = CONFIG.expanded_feature_sets[0][0]
    (output / name).mkdir(parents=True)
    (output / name / "manifest.json").write_text("{}")

    with pytest.raises(RuntimeError, match="partial variant"):
        expanded.build_variants(tmp_path / "baseline.yaml", tmp_path / "source.yaml", output)


def test_fit_pins_manifest_and_restricts_one_model_fold(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    matrix_root, results_root = tmp_path / "matrices", tmp_path / "results"
    name, additions = CONFIG.expanded_feature_sets[2]
    matrix = _complete_variant(matrix_root, name, additions)
    expected_sha = hashlib.sha256((matrix / "manifest.json").read_bytes()).hexdigest()
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: "cpu-node-1")
    monkeypatch.setattr(expanded, "run_remote_fits", calls.append)

    expanded.fit_variant(name, matrix_root, results_root, 2)

    assert len(calls) == 1
    config = calls[0]
    assert config.processed_matrix_root == matrix
    assert config.processed_matrix_manifest_sha256 == expected_sha
    assert config.results_dir == results_root / name
    assert config.fit_backends == ("frequentist",)
    assert config.execution_mode == "execute"
    assert config.execution_target == "swc_hpc"
    assert config.resume_completed_folds
    assert config.model_folds == (CONFIG.model_folds[2],)
    assert config.item_folds == 4
    assert set(config.feature_blocks) >= {
        *CONFIG.feature_blocks,
        "surprisal-llama", "surprisal-position-adjusted", "embedding-sae-shared",
    }


@pytest.mark.parametrize("name,additions", CONFIG.expanded_feature_sets)
def test_fit_keeps_variant_results_separate(
    tmp_path: Path, monkeypatch, name: str, additions: tuple[str, ...],
) -> None:
    expanded = _expanded()
    matrix_root, results_root = tmp_path / "matrices", tmp_path / "results"
    _complete_variant(matrix_root, name, additions)
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: "cpu-node-1")
    monkeypatch.setattr(expanded, "run_remote_fits", calls.append)

    expanded.fit_variant(name, matrix_root, results_root, 0)

    assert calls[0].results_dir == results_root / name


def test_fit_requires_slurm_before_touching_the_matrix(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: "cpu-node-1")
    monkeypatch.setattr(
        expanded, "_resume_variant", lambda *_: pytest.fail("guard touched matrix"),
    )

    with pytest.raises(RuntimeError, match="Slurm allocation"):
        expanded.fit_variant("readable-surprisal", tmp_path, tmp_path / "results", 0)


def test_fit_refuses_gateway_before_touching_the_matrix(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: "sgw2")
    monkeypatch.setattr(
        expanded, "_resume_variant", lambda *_: pytest.fail("guard touched matrix"),
    )

    with pytest.raises(RuntimeError, match="SWC gateway"):
        expanded.fit_variant("readable-surprisal", tmp_path, tmp_path / "results", 0)


def test_build_cli_dispatches_paths(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    calls = []
    paths = [tmp_path / name for name in ("baseline.yaml", "canonical.yaml", "matrices")]
    monkeypatch.setattr(expanded, "build_variants", lambda *args: calls.append(args))

    expanded.main(["build", *map(str, paths)])

    assert calls == [tuple(paths)]


def test_fit_cli_dispatches_variant_paths_and_fold(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    calls = []
    matrix_root, results_root = tmp_path / "matrices", tmp_path / "results"
    monkeypatch.setattr(expanded, "fit_variant", lambda *args: calls.append(args))

    expanded.main([
        "fit", "readable-sae", str(matrix_root), str(results_root), "3",
    ])

    assert calls == [("readable-sae", matrix_root, results_root, 3)]


def test_expanded_sbatch_has_valid_bash_syntax() -> None:
    script = Path(__file__).with_name("expanded.sbatch")

    result = subprocess.run(["bash", "-n", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
