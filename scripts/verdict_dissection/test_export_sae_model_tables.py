import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).parent))

KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]


def test_tables_pool_aime_scale_on_development_and_build_left_minus_right_pairs():
    module = importlib.import_module("export_sae_model_tables")
    transcripts = pd.DataFrame([
        ["aime2024", "a", "q1", 0, "la", 1, "ha", "representation_train", 1.0, 0.0, 9.0, 99.0],
        ["aime2024", "b", "q1", 0, "lb", 1, "hb", "representation_train", 3.0, 2.0, 8.0, 98.0],
        ["aime2025", "c", "q2", 0, "lc", 1, "hc", "representation_validation", 5.0, 0.0, 7.0, 97.0],
        ["aime2026", "d", "q3", 0, "ld", 1, "hd", "confirmation", 7.0, 4.0, 6.0, 96.0],
    ], columns=KEYS + ["split_role", "sae_shared_0", "sae_shared_1", "sae_aime_0", "qwen_dim_0000"])
    verdicts = pd.DataFrame([
        ["aime2024", "j", "q1", "a", "b", 0, 0, "la", "lb", True],
        ["aime2024", "j", "q1", "b", "a", 0, 0, "lb", "la", False],
    ], columns=["bench", "judge", "item", "model_left", "model_right", "epoch_left",
                "epoch_right", "log_left", "log_right", "pick_left"])

    tables = module.build_tables(transcripts, verdicts)
    raw = tables["transcript/raw.parquet"]
    normalised = tables["transcript/normalised.parquet"]
    params = tables["normalisation-parameters.parquet"]
    pairs = tables["pair/raw.parquet"]
    normalised_pairs = tables["pair/normalised.parquet"]

    assert raw.bench_group.tolist() == ["aime"] * 4
    assert raw.sae_shared_1_fired.tolist() == [0, 1, 0, 1]
    assert "sae_aime_0" not in raw and "qwen_dim_0000" not in raw
    assert normalised.sae_shared_0.tolist() == pytest.approx([-1.0, 0.0, 1.0, 2.0])
    assert params.loc[params.feature == "sae_shared_0", ["benchmark", "mean", "sd"]].iloc[0].tolist() == ["aime", 3.0, 2.0]
    assert pairs[["sae_shared_0_left", "sae_shared_0_right", "d_sae_shared_0"]].values.tolist() == [[1.0, 3.0, -2.0], [3.0, 1.0, 2.0]]
    assert pairs.d_sae_shared_1_fired.tolist() == [-1, 1]
    assert normalised_pairs.d_sae_shared_0.tolist() == pytest.approx([-1.0, 1.0])
    assert tables["feature-dictionary.parquet"].feature.tolist() == [
        "sae_shared_0", "sae_shared_1", "sae_shared_0_fired", "sae_shared_1_fired"]


def test_unknown_split_role_is_rejected():
    module = importlib.import_module("export_sae_model_tables")
    transcripts = pd.DataFrame([
        ["gpqa", "a", "q1", 0, "la", 1, "ha", "future_split", 1.0],
    ], columns=KEYS + ["split_role", "sae_shared_0"])
    verdicts = pd.DataFrame(columns=["bench", "item"])

    with pytest.raises(ValueError, match="unknown split_role"):
        module.build_tables(transcripts, verdicts)
