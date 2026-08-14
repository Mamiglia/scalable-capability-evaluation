import os
import pathlib
import wbconfig


def test_cfg_loads_and_resolves():
    c = wbconfig.cfg()
    assert c["models"]["dev"] == "Qwen3-0.6B"
    assert c["paths"]["pairs"].endswith("pairs.parquet")
    assert c["screen"]["batch_tokens"] > 0


def test_paths_out_is_absolute():
    c = wbconfig.cfg()
    out_path = c["paths"]["out"]
    assert pathlib.Path(out_path).is_absolute(), f"out path must be absolute: {out_path}"
    assert out_path.endswith("judge-whitebox/out"), f"out path must end with judge-whitebox/out: {out_path}"


def test_hpc_env_selects_ceph_paths(monkeypatch):
    monkeypatch.setenv("WB_HPC", "1")
    c = wbconfig.cfg()
    pairs_path = c["paths"]["pairs"]
    assert pairs_path.startswith("/ceph/"), f"HPC pairs path must start with /ceph/: {pairs_path}"
    assert pairs_path.endswith("pairs.parquet")


def test_wb_pairs_env_override(monkeypatch, tmp_path):
    test_pairs_file = tmp_path / "custom_pairs.parquet"
    test_pairs_file.touch()
    monkeypatch.setenv("WB_PAIRS", str(test_pairs_file))
    c = wbconfig.cfg()
    assert c["paths"]["pairs"] == str(test_pairs_file)
    assert pathlib.Path(c["paths"]["pairs"]).is_absolute()


def test_resolve_model_uses_hpc_dir_under_wb_hpc(monkeypatch):
    monkeypatch.setenv("WB_HPC", "1")
    c = wbconfig.cfg()
    assert wbconfig.resolve_model("Qwen3-8B", c) == f"{c['models']['hpc_dir']}/Qwen3-8B"


def test_resolve_model_uses_hf_id_without_wb_hpc(monkeypatch):
    monkeypatch.delenv("WB_HPC", raising=False)
    c = wbconfig.cfg()
    assert wbconfig.resolve_model("Qwen3-8B", c) == "Qwen/Qwen3-8B"
