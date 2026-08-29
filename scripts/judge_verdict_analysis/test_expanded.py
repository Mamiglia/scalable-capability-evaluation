from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import yaml

from .build_dm import ARTIFACTS
from .config import CONFIG


def _expanded():
    return importlib.import_module("scripts.judge_verdict_analysis.expanded")


BASELINE_FEATURES = tuple(f"readable_{index:03d}" for index in range(117))
GROUPED_OUT_FEATURES = tuple(f"grouped_out_{index:02d}" for index in range(20))


def _allow_swc(monkeypatch, expanded) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: "cpu-node-1")


def _representative_map(additions: tuple[str, ...] = ()) -> pd.DataFrame:
    features = BASELINE_FEATURES + GROUPED_OUT_FEATURES + additions
    retained = [True] * 117 + [False] * 20 + [True] * len(additions)
    return pd.DataFrame({
        "feature": features,
        "status": ["retained"] * 117 + ["grouped_out"] * 20 + ["retained"] * len(additions),
        "correlation_group": [pd.NA] * len(features),
        "representative": [
            *BASELINE_FEATURES,
            *([BASELINE_FEATURES[0]] * 20),
            *additions,
        ],
        "retained": retained,
        "original_block": ["surface"] * len(features),
        "group_size": [1] * 117 + [2] * 20 + [1] * len(additions),
        "processed_name": [
            *BASELINE_FEATURES,
            *([BASELINE_FEATURES[0]] * 20),
            *additions,
        ],
    })


def _write_matrix(root: Path, features: tuple[str, ...]) -> None:
    root.mkdir(parents=True)
    pd.DataFrame({
        "feature": features,
        "block": ["surface"] * len(features),
        "source": ["fixture"] * len(features),
        "transform": ["none"] * len(features),
        "n_missing": [0] * len(features),
        "correlation_group": [pd.NA] * len(features),
        "representative": features,
        "retained": [True] * len(features),
        "original_block": ["surface"] * len(features),
        "group_size": [1] * len(features),
        "processed_name": features,
    }).to_parquet(root / "feature-dictionary.parquet", index=False)
    pd.DataFrame({
        "bench": ["b"], "model": ["m"], "item": ["i"],
        "split": ["representation_train"], "model_accuracy": [1.0],
        **{feature: [0.0] for feature in features},
    }).to_parquet(root / "transcript.parquet", index=False)
    pd.DataFrame({
        "bench": ["b"], "judge": ["j"], "item": ["i"],
        "model_left": ["m1"], "model_right": ["m2"],
        "split": ["representation_train"], "pick_left": [1],
        **{f"d_{feature}": [0.0] for feature in features},
    }).to_parquet(root / "pair.parquet", index=False)
    for filename, prefix in (
        ("scaling-transcript.parquet", ""), ("scaling-pair.parquet", "d_"),
    ):
        pd.DataFrame({
            "benchmark": ["b"] * len(features),
            "feature": [f"{prefix}{feature}" for feature in features],
            "mean": [0.0] * len(features), "sd": [1.0] * len(features),
        }).to_parquet(root / filename, index=False)


def _source_fixture(tmp_path: Path, monkeypatch, expanded) -> tuple[Path, Path]:
    baseline_root, source_root = tmp_path / "baseline", tmp_path / "source"
    additions = tuple(dict.fromkeys(
        feature for _, features in CONFIG.expanded_feature_sets for feature in features
    ))
    _write_matrix(baseline_root, BASELINE_FEATURES)
    _write_matrix(source_root, BASELINE_FEATURES + additions)
    _representative_map().to_csv(baseline_root / "representative-map.csv", index=False)
    (baseline_root / "manifest.json").write_text(json.dumps({
        "source_hf_repo": "example/judging",
        "source_hf_revision": "a" * 40,
        "source_status": "resolved",
    }))
    baseline_config, source_config = tmp_path / "baseline.yaml", tmp_path / "source.yaml"
    baseline_config.write_text(yaml.safe_dump({"output_root": str(baseline_root)}))
    source_config.write_text(yaml.safe_dump({"output_root": str(source_root)}))
    pin = hashlib.sha256((baseline_root / "manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(
        expanded, "CONFIG", replace(CONFIG, processed_matrix_manifest_sha256=pin),
    )
    return baseline_config, source_config


def _source_paths(baseline_config: Path, source_config: Path) -> dict[str, Path]:
    baseline_root = Path(yaml.safe_load(baseline_config.read_text())["output_root"])
    source_root = Path(yaml.safe_load(source_config.read_text())["output_root"])
    paths = {
        "baseline_config": baseline_config,
        "canonical_config": source_config,
        "baseline_manifest": baseline_root / "manifest.json",
        "baseline_representative_map": baseline_root / "representative-map.csv",
    }
    for prefix, root in (("baseline", baseline_root), ("canonical", source_root)):
        for filename in ARTIFACTS[:5]:
            key = f"{prefix}_{filename.removesuffix('.parquet').replace('-', '_')}"
            paths[key] = root / filename
    return paths


def test_build_materializes_the_three_exact_variants(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    calls = []
    def fake_build(baseline_config, source_config, additions, output):
        calls.append((
            Path(yaml.safe_load(baseline_config.read_text())["output_root"]),
            Path(yaml.safe_load(source_config.read_text())["output_root"]),
            tuple(additions), output,
        ))
        return {"manifest.json": {
            "source_paths": {
                "baseline_config": str(baseline_config),
                "canonical_config": str(source_config),
            },
            "source_sha256": {"baseline_config": "", "canonical_config": ""},
        }}

    monkeypatch.setattr(expanded, "build_expanded_design_matrix", fake_build)
    monkeypatch.setattr(expanded, "write_design_matrix", lambda tables, output: None)
    output = tmp_path / "matrices"

    expanded.build_variants(baseline, source, output)

    assert [call[2] for call in calls] == [features for _, features in CONFIG.expanded_feature_sets]
    assert [len(call[2]) for call in calls] == [10, 32, 42]
    assert [call[3] for call in calls] == [output / name for name, _ in CONFIG.expanded_feature_sets]
    assert all(path.is_absolute() for call in calls for path in call[:2])


def test_build_normalizes_relative_yaml_roots_end_to_end(
    tmp_path: Path, monkeypatch,
) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    baseline.write_text(yaml.safe_dump({"output_root": "baseline"}))
    source.write_text(yaml.safe_dump({"output_root": "source"}))
    output_root = tmp_path / "matrices"

    expanded.build_variants(baseline, source, output_root)

    for name, additions in CONFIG.expanded_feature_sets:
        output = output_root / name
        manifest = json.loads((output / "manifest.json").read_text())
        assert all(Path(path).is_absolute() for path in manifest["source_paths"].values())
        assert expanded._resume_variant(output, additions, baseline, source)


def test_build_resolves_production_yaml_roots_from_repository_cwd(
    tmp_path: Path, monkeypatch,
) -> None:
    expanded = _expanded()
    repo = tmp_path / "repo"
    repo.mkdir()
    _allow_swc(monkeypatch, expanded)
    baseline_config, source_config = _source_fixture(repo, monkeypatch, expanded)
    baseline_root = repo / "judge_results/capability/design-matrix"
    source_root = repo / "judge_results/canonical/design-matrix"
    baseline_root.parent.mkdir(parents=True)
    source_root.parent.mkdir(parents=True)
    (repo / "baseline").rename(baseline_root)
    (repo / "source").rename(source_root)
    baseline = baseline_root / "design-matrix.yaml"
    source = source_root / "design-matrix.yaml"
    baseline_config.rename(baseline)
    source_config.rename(source)
    baseline.write_text(yaml.safe_dump({
        "output_root": "judge_results/capability/design-matrix",
    }))
    source.write_text(yaml.safe_dump({
        "output_root": "judge_results/canonical/design-matrix",
    }))
    monkeypatch.chdir(repo)
    output_root = repo / "expanded"

    expanded.build_variants(baseline, source, output_root)

    for name, additions in CONFIG.expanded_feature_sets:
        assert expanded._resume_variant(output_root / name, additions, baseline, source)


def test_relative_yaml_root_rejects_missing_candidates(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    cwd, config_dir = tmp_path / "repo", tmp_path / "configs"
    cwd.mkdir()
    config_dir.mkdir()
    config = config_dir / "matrix.yaml"
    config.write_text(yaml.safe_dump({"output_root": "missing"}))
    monkeypatch.chdir(cwd)

    with pytest.raises(RuntimeError, match="does not resolve"):
        expanded._config_root(config)


def test_relative_yaml_root_rejects_ambiguous_candidates(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    cwd, config_dir = tmp_path / "repo", tmp_path / "configs"
    (cwd / "matrix").mkdir(parents=True)
    (config_dir / "matrix").mkdir(parents=True)
    config = config_dir / "matrix.yaml"
    config.write_text(yaml.safe_dump({"output_root": "matrix"}))
    monkeypatch.chdir(cwd)

    with pytest.raises(RuntimeError, match="ambiguous"):
        expanded._config_root(config)


def _complete_variant(
    root: Path,
    name: str,
    additions: tuple[str, ...],
    baseline_config: Path,
    source_config: Path,
) -> Path:
    output = root / name
    inventory = [*BASELINE_FEATURES, *additions]
    _write_matrix(output, tuple(inventory))
    _representative_map(additions).to_csv(output / "representative-map.csv", index=False)
    source_paths = _source_paths(baseline_config, source_config)
    manifest = {
        "source_hf_repo": "example/judging",
        "source_hf_revision": "a" * 40,
        "source_status": "resolved",
        "source_paths": {key: str(path) for key, path in source_paths.items()},
        "source_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in source_paths.items()
        },
        "settings": {
            "construction": "processed-baseline-plus-explicit-additions",
            "additions": list(additions),
            "correlation_grouping": False,
        },
        "features": {
            "baseline": 117,
            "added": len(additions),
            "retained": len(inventory),
            "inventory": inventory,
            "additions": list(additions),
        },
        "rows": {"transcript": 1, "pair": 1},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "design-matrix.yaml").write_text(yaml.safe_dump({
        "output_root": str(output),
        "source_hf_repo": manifest["source_hf_repo"],
        "source_hf_revision": manifest["source_hf_revision"],
        "source_status": manifest["source_status"],
    }))
    return output


def test_build_resumes_only_complete_matching_variants(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    output = tmp_path / "matrices"
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    for name, additions in CONFIG.expanded_feature_sets:
        _complete_variant(output, name, additions, baseline, source)
    monkeypatch.setattr(
        expanded, "build_expanded_design_matrix",
        lambda *_: pytest.fail("complete variant was rebuilt"),
    )

    expanded.build_variants(baseline, source, output)


def test_build_refuses_complete_variants_from_stale_inputs(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    output = tmp_path / "matrices"
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    for name, additions in CONFIG.expanded_feature_sets:
        _complete_variant(output, name, additions, baseline, source)
    baseline_root = Path(yaml.safe_load(baseline.read_text())["output_root"])
    (baseline_root / "representative-map.csv").write_text("changed")

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded.build_variants(baseline, source, output)


def test_build_refuses_a_partial_existing_variant(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    output = tmp_path / "matrices"
    name = CONFIG.expanded_feature_sets[0][0]
    (output / name).mkdir(parents=True)
    (output / name / "manifest.json").write_text("{}")

    with pytest.raises(RuntimeError, match="partial variant"):
        expanded.build_variants(baseline, source, output)


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_resume_reads_and_rejects_each_corrupt_artifact(
    tmp_path: Path, monkeypatch, artifact: str,
) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    if artifact.endswith(".parquet"):
        (output / artifact).write_bytes(b"not parquet")
    else:
        (output / artifact).write_text("{broken")

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_manifest_row_count_mismatch(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["rows"]["pair"] = 2
    (output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_scaling_missing_one_benchmark_feature(
    tmp_path: Path, monkeypatch,
) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    for filename in ("transcript.parquet", "pair.parquet"):
        table = pd.read_parquet(output / filename)
        second = table.assign(bench="b2", item="i2")
        pd.concat([table, second], ignore_index=True).to_parquet(output / filename, index=False)
    for filename in ("scaling-transcript.parquet", "scaling-pair.parquet"):
        scaling = pd.read_parquet(output / filename)
        second = scaling.assign(benchmark="b2").iloc[1:]
        pd.concat([scaling, second], ignore_index=True).to_parquet(output / filename, index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["rows"] = {"transcript": 2, "pair": 2}
    (output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_feature_inventory_mismatch(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    dictionary = pd.read_parquet(output / "feature-dictionary.parquet")
    dictionary.loc[0, "feature"] = "wrong"
    dictionary.to_parquet(output / "feature-dictionary.parquet", index=False)

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_retained_provenance_outside_inventory(
    tmp_path: Path, monkeypatch,
) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    representative_map = pd.read_csv(output / "representative-map.csv")
    representative_map.loc[
        representative_map.feature == GROUPED_OUT_FEATURES[0], ["retained", "status"]
    ] = [True, "retained"]
    representative_map.to_csv(output / "representative-map.csv", index=False)

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_wrong_retained_provenance_order(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    representative_map = pd.read_csv(output / "representative-map.csv")
    representative_map = representative_map.iloc[
        [1, 0, *range(2, len(representative_map))]
    ].reset_index(drop=True)
    representative_map.to_csv(output / "representative-map.csv", index=False)

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


@pytest.mark.parametrize("column,bad_value", [
    ("retained", False),
    ("status", "grouped_out"),
    ("representative", "wrong"),
    ("processed_name", "wrong"),
])
def test_resume_rejects_corrupt_addition_provenance(
    tmp_path: Path, monkeypatch, column: str, bad_value: object,
) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    representative_map = pd.read_csv(output / "representative-map.csv")
    representative_map.loc[representative_map.feature == additions[0], column] = bad_value
    representative_map.to_csv(output / "representative-map.csv", index=False)

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_missing_pair_metadata(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    pair = pd.read_parquet(output / "pair.parquet").drop(columns="judge")
    pair.to_parquet(output / "pair.parquet", index=False)

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_resume_rejects_incomplete_provenance_key_set(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    name, additions = CONFIG.expanded_feature_sets[0]
    output = _complete_variant(tmp_path / "matrices", name, additions, baseline, source)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["source_paths"].pop("canonical_pair")
    manifest["source_sha256"].pop("canonical_pair")
    (output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="invalid variant"):
        expanded._resume_variant(output, additions, baseline, source)


def test_build_rejects_a_noncanonical_baseline_feature_count(
    tmp_path: Path, monkeypatch,
) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    baseline_root = Path(yaml.safe_load(baseline.read_text())["output_root"])
    dictionary = pd.read_parquet(baseline_root / "feature-dictionary.parquet").head(1)
    dictionary.to_parquet(baseline_root / "feature-dictionary.parquet", index=False)
    monkeypatch.setattr(
        expanded, "build_expanded_design_matrix",
        lambda *_: pytest.fail("unauthorized baseline reached matrix construction"),
    )

    with pytest.raises(RuntimeError, match="exactly 117 features"):
        expanded.build_variants(baseline, source, tmp_path / "matrices")


def test_build_rejects_an_unpinned_baseline_manifest(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    monkeypatch.setattr(
        expanded, "CONFIG",
        replace(expanded.CONFIG, processed_matrix_manifest_sha256="0" * 64),
    )

    with pytest.raises(RuntimeError, match="manifest SHA"):
        expanded.build_variants(baseline, source, tmp_path / "matrices")


def test_build_rejects_missing_canonical_additions(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    _allow_swc(monkeypatch, expanded)
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    source_root = Path(yaml.safe_load(source.read_text())["output_root"])
    dictionary = pd.read_parquet(source_root / "feature-dictionary.parquet")
    dictionary = dictionary[dictionary.feature != CONFIG.expanded_feature_sets[0][1][0]]
    dictionary.to_parquet(source_root / "feature-dictionary.parquet", index=False)

    with pytest.raises(RuntimeError, match="canonical dictionary additions"):
        expanded.build_variants(baseline, source, tmp_path / "matrices")


def test_fit_pins_manifest_and_restricts_one_model_fold(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    matrix_root, results_root = tmp_path / "matrices", tmp_path / "results"
    name, additions = CONFIG.expanded_feature_sets[2]
    matrix = _complete_variant(matrix_root, name, additions, baseline, source)
    expected_sha = hashlib.sha256((matrix / "manifest.json").read_bytes()).hexdigest()
    calls = []
    _allow_swc(monkeypatch, expanded)
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
    baseline, source = _source_fixture(tmp_path, monkeypatch, expanded)
    matrix_root, results_root = tmp_path / "matrices", tmp_path / "results"
    _complete_variant(matrix_root, name, additions, baseline, source)
    calls = []
    _allow_swc(monkeypatch, expanded)
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


@pytest.mark.parametrize("hostname,job_id,message", [
    ("cpu-node-1", None, "Slurm allocation"),
    ("gateway-1", "123", "SWC gateway"),
])
def test_build_guards_before_loading_parquet(
    tmp_path: Path, monkeypatch, hostname: str, job_id: str | None, message: str,
) -> None:
    expanded = _expanded()
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.setattr(expanded.socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(
        expanded.pd, "read_parquet", lambda *_: pytest.fail("guard loaded Parquet"),
    )

    with pytest.raises(RuntimeError, match=message):
        expanded.build_variants(
            tmp_path / "baseline.yaml", tmp_path / "source.yaml", tmp_path / "matrices",
        )


def test_build_cli_dispatches_paths(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    calls = []
    monkeypatch.chdir(tmp_path)
    names = ("baseline.yaml", "canonical.yaml", "matrices")
    monkeypatch.setattr(expanded, "build_variants", lambda *args: calls.append(args))

    expanded.main(["build", *names])

    assert calls == [tuple(tmp_path / name for name in names)]


def test_fit_cli_dispatches_variant_paths_and_fold(tmp_path: Path, monkeypatch) -> None:
    expanded = _expanded()
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(expanded, "fit_variant", lambda *args: calls.append(args))

    expanded.main([
        "fit", "readable-sae", "matrices", "results", "3",
    ])

    assert calls == [("readable-sae", tmp_path / "matrices", tmp_path / "results", 3)]


def test_expanded_sbatch_has_valid_bash_syntax() -> None:
    script = Path(__file__).with_name("expanded.sbatch")

    result = subprocess.run(["bash", "-n", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_expanded_sbatch_rejects_relative_bundle_paths(tmp_path: Path) -> None:
    script = Path(__file__).with_name("expanded.sbatch")
    result = subprocess.run(
        ["bash", script, "build", "baseline.yaml", "canonical.yaml", "matrices"],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "SLURM_JOB_ID": "1",
            "SLURM_SUBMIT_DIR": str(tmp_path),
            "JV_CODE_DIR": str(Path(__file__).resolve().parents[2]),
            "JV_PYTHON": "/usr/bin/true",
            "JV_PACKAGES": str(tmp_path),
        },
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 2
    assert "absolute" in result.stderr
