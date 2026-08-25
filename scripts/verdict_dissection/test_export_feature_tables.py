"""Contract checks for the portable verdict-dissection table export."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from export_feature_tables import export_tables


KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]


def test_export_writes_canonical_tables_with_accuracy_and_local_scaling(tmp_path):
    # This catches accidentally counting incorrect or non-canonical transcripts.
    store = pd.DataFrame([
        ["a", "m1", "q1", 0, "l1", 1, "h1", True, True, "kept"],
        ["a", "m1", "q2", 0, "l1", 2, "h2", False, True, "kept"],
        ["b", "m2", "q3", 0, "l2", 1, "h3", True, True, "kept"],
        ["a", "m1", "old", 0, "old", 1, "old", True, False, "discarded"],
        ["a", "m1", "q1", 0, "l1", 2, "h1-later", True, True, "later duplicate"],
    ], columns=KEYS + ["correct", "is_canonical", "provenance"])
    raw = pd.DataFrame([
        ["a", "m1", "q1", 0, "l1", 1, "h1", 10.0, 1.0, "text"],
        ["a", "m1", "q2", 0, "l1", 2, "h2", 20.0, 3.0, "text"],
        ["b", "m2", "q3", 0, "l2", 1, "h3", 30.0, 5.0, "text"],
        ["a", "m1", "q1", 0, "l1", 2, "h1-later", 99.0, 9.0, "later"],
    ], columns=KEYS + ["n_words", "surprisal_full__tiny", "label"])
    verdicts = pd.DataFrame([{
        "bench": "a", "item": "q1", "model_left": "m1", "model_right": "m1",
        "epoch_left": 0, "epoch_right": 0, "log_left": "l1", "log_right": "l1",
        "pick_left": 1,
    }, {
        "bench": "a", "item": "missing", "model_left": "m1", "model_right": "m1",
        "epoch_left": 0, "epoch_right": 0, "log_left": "l1", "log_right": "l1",
        "pick_left": 1,
    }])

    export_tables(store, verdicts, raw, tmp_path, content_hash="content", contract_hash="contract")

    expected = {
        "raw/identity.parquet", "raw/model-benchmark-accuracy.parquet", "raw/surface-textual.parquet",
        "raw/context-predictability.parquet", "raw/transcript-features.parquet",
        "normalised/transcript-features-global.parquet",
        "normalised/transcript-features-within-benchmark.parquet",
        "normalised/normalisation-parameters.parquet", "paired/judging-rounds.parquet",
        "paired/raw-features.parquet", "paired/normalised-features.parquet",
        "feature-dictionary.parquet", "manifest.parquet",
    }
    assert all((tmp_path / path).exists() for path in expected)
    assert len(pd.read_parquet(tmp_path / "raw" / "identity.parquet")) == 5
    accuracy = pd.read_parquet(tmp_path / "raw" / "model-benchmark-accuracy.parquet")
    assert accuracy.loc[0, ["correct_n", "total_n", "accuracy"]].tolist() == [2, 3, 2 / 3]
    assert pd.read_parquet(tmp_path / "raw" / "surface-textual.parquet").columns.tolist() == KEYS + ["n_words"]
    assert pd.read_parquet(tmp_path / "raw" / "context-predictability.parquet").columns.tolist() == KEYS + ["surprisal_full__tiny"]
    transcript = pd.read_parquet(tmp_path / "raw" / "transcript-features.parquet")
    assert len(transcript) == 3 and transcript.correct.tolist() == [True, True, True]
    within = pd.read_parquet(tmp_path / "normalised" / "transcript-features-within-benchmark.parquet")
    assert within.loc[within.bench == "b", "n_words"].iloc[0] == 0.0
    assert "label" in within and within.label.tolist() == ["text", "text", "later"]
    params = pd.read_parquet(tmp_path / "normalised" / "normalisation-parameters.parquet")
    assert set(params.scope) == {"global", "within_benchmark"}
    dictionary = pd.read_parquet(tmp_path / "feature-dictionary.parquet")
    assert dictionary.feature.tolist() == ["n_words", "surprisal_full__tiny"]
    assert dictionary.group.tolist() == ["surface-textual", "context-predictability"]
    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert manifest.loc[0, ["transcript_rows", "judging_rounds", "raw_pair_rows"]].tolist() == [3, 2, 1]
    assert manifest.loc[0, ["content_hash", "contract_hash"]].tolist() == ["content", "contract"]
    pairs = pd.read_parquet(tmp_path / "paired" / "raw-features.parquet")
    assert len(pairs) == 1 and pairs.loc[0, "d_n_words"] == 0.0
    assert pairs.loc[0, "n_words_left"] == 10.0
    assert {"model_left", "model_right", "d_surprisal_full__tiny"} <= set(pairs)
    assert len(pd.read_parquet(tmp_path / "paired" / "judging-rounds.parquet")) == 2
