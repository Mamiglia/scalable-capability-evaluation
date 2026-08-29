import json
import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from capability_predictions_glm.config import CONFIG, SOURCE_MANIFEST_SHA256, SOURCE_NAME
from capability_predictions_glm.data import feature_variants, load_inputs, validate_manifest
from capability_predictions_glm.preprocessing import (
    attach_direct_target,
    attach_pair_target,
    deduplicate_feature_pairs,
    item_quartiles,
    pool_aime,
    quartile_accuracy,
    split_pairs,
    split_transcripts,
)


def test_feature_variants_separate_llm_annotations() -> None:
    dictionary = pd.DataFrame({
        "feature": ["length", "rigour", "error_count", "density"],
        "block": ["surface", "rubric", "intermediate-errors", "clause-descriptors"],
    })

    variants = feature_variants(dictionary)

    assert variants == {
        "no_llm_annotations": ("length", "density"),
        "with_llm_annotations": ("length", "rigour", "error_count", "density"),
    }


def test_manifest_must_pin_old_annotated_subset(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "source": {"name": SOURCE_NAME, "design_matrix_manifest_sha256": SOURCE_MANIFEST_SHA256}
    }))

    assert validate_manifest(manifest)["source"]["name"] == SOURCE_NAME

    manifest.write_text(json.dumps({"source": {"name": "wrong", "design_matrix_manifest_sha256": "wrong"}}))
    with pytest.raises(ValueError, match="wrong capability input subset"):
        validate_manifest(manifest)


def test_load_inputs_checks_hashes_and_feature_columns(tmp_path) -> None:
    tables = {
        "feature-dictionary.parquet": pd.DataFrame({"feature": ["x", "quality"],
                                                     "block": ["surface", "rubric"]}),
        "responses.parquet": pd.DataFrame({
            "bench": ["gpqa"], "model": ["m"], "item": ["i"],
            "correct": [1], "is_canonical": [True],
        }),
        "transcript.parquet": pd.DataFrame({
            "bench": ["gpqa"], "model": ["m"], "item": ["i"],
            "epoch": [0], "log": ["l"], "x": [0.1], "quality": [1.0],
        }),
        "pair.parquet": pd.DataFrame({
            "bench": ["gpqa"], "judge": ["j"], "item": ["i"],
            "model_left": ["m"], "model_right": ["n"], "pick_left": [1],
            "d_x": [0.2], "d_quality": [1.0],
        }),
    }
    files = {}
    for name, table in tables.items():
        path = tmp_path / name
        table.to_parquet(path, index=False)
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source": {"name": SOURCE_NAME, "design_matrix_manifest_sha256": SOURCE_MANIFEST_SHA256},
        "files": files,
    }))

    inputs = load_inputs(replace(CONFIG, input_root=tmp_path))

    assert inputs.variants["no_llm_annotations"] == ("x",)
    assert inputs.transcripts.columns.tolist()[-2:] == ["x", "quality"]
    assert inputs.pairs.columns.tolist()[-2:] == ["d_x", "d_quality"]

    tables_path = tmp_path / "responses.parquet"
    tables_path.write_bytes(tables_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_inputs(replace(CONFIG, input_root=tmp_path))


def test_item_quartiles_are_fixed_by_equal_model_solve_rate() -> None:
    rates = {
        "i1": (1, 1, 1, 1),
        "i2": (1, 1, 1, 0),
        "i3": (1, 1, 0, 0),
        "i4": (1, 0, 1, 0),
        "i5": (1, 0, 0, 0),
        "i6": (0, 1, 0, 0),
        "i7": (0, 0, 0, 0),
        "i8": (0, 0, 0, 0),
    }
    rows = [
        {"bench": "gpqa", "model": f"m{model}", "item": item,
         "correct": correct, "is_canonical": True}
        for item, outcomes in rates.items()
        for model, correct in enumerate(outcomes, 1)
    ]
    # A repeated attempt must not give m1 more weight than another model.
    rows.append({"bench": "gpqa", "model": "m1", "item": "i3",
                 "correct": 0, "is_canonical": True})

    quartiles = item_quartiles(pd.DataFrame(rows))

    assert quartiles.groupby("quartile").item.apply(list).to_dict() == {
        "Q1": ["i1", "i2"],
        "Q2": ["i4", "i3"],
        "Q3": ["i5", "i6"],
        "Q4": ["i7", "i8"],
    }
    assert quartiles.set_index("item").loc["i3", "solve_rate"] == pytest.approx(0.375)


def test_pool_aime_keeps_year_items_distinct() -> None:
    rows = pd.DataFrame({
        "bench": ["aime2024", "aime2025", "gpqa"],
        "item": ["1", "1", "x"],
    })

    pooled = pool_aime(rows)

    assert pooled[["bench", "item"]].to_records(index=False).tolist() == [
        ("aime", "aime2024/1"), ("aime", "aime2025/1"), ("gpqa", "x")]


def test_quartile_accuracy_weights_items_not_attempts() -> None:
    responses = pd.DataFrame([
        {"bench": "b", "model": "m", "item": "i1", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "m", "item": "i1", "correct": 0, "is_canonical": True},
        {"bench": "b", "model": "m", "item": "i2", "correct": 1, "is_canonical": True},
    ])
    quartiles = pd.DataFrame({"bench": ["b", "b"], "item": ["i1", "i2"],
                              "quartile": ["Q2", "Q2"]})

    result = quartile_accuracy(responses, quartiles)

    assert result.loc[0, "accuracy"] == pytest.approx(0.75)


def test_outer_splits_use_q1_and_hold_out_whole_models() -> None:
    quartiles = pd.DataFrame({
        "bench": ["b", "b"], "item": ["easy", "hard"], "quartile": ["Q1", "Q4"]})
    transcripts = pd.DataFrame([
        {"bench": "b", "item": item, "model": model, "x": 1.0}
        for item in ("easy", "hard") for model in ("seen", "held")
    ])
    pairs = pd.DataFrame([
        {"bench": "b", "item": "easy", "model_left": "seen", "model_right": "seen2"},
        {"bench": "b", "item": "easy", "model_left": "held", "model_right": "seen"},
        {"bench": "b", "item": "hard", "model_left": "held", "model_right": "seen"},
    ])

    train_t, test_t = split_transcripts(transcripts, quartiles, {"held"})
    train_p, test_p = split_pairs(pairs, quartiles, {"held"})

    assert train_t[["item", "model"]].to_records(index=False).tolist() == [("easy", "seen")]
    assert test_t[["item", "model"]].to_records(index=False).tolist() == [("easy", "held")]
    assert train_p[["model_left", "model_right"]].to_records(index=False).tolist() == [("seen", "seen2")]
    assert test_p[["model_left", "model_right"]].to_records(index=False).tolist() == [("held", "seen")]


def test_targets_use_requested_quartile_and_feature_pairs_drop_judge_duplicates() -> None:
    accuracy = pd.DataFrame([
        {"bench": "b", "model": "a", "quartile": "Q2", "accuracy": 0.8},
        {"bench": "b", "model": "b", "quartile": "Q2", "accuracy": 0.3},
        {"bench": "b", "model": "a", "quartile": "Q3", "accuracy": 0.1},
        {"bench": "b", "model": "b", "quartile": "Q3", "accuracy": 0.9},
    ])
    transcripts = pd.DataFrame([
        {"bench": "b", "model": "a", "item": "easy"},
        {"bench": "b", "model": "b", "item": "easy"},
    ])
    pairs = pd.DataFrame([
        {"bench": "b", "judge": judge, "item": "easy", "model_left": "a", "model_right": "b",
         "epoch_left": 0, "epoch_right": 0, "log_left": "la", "log_right": "lb", "d_x": 1.0,
         "accuracy_left": 0.0, "accuracy_right": 1.0, "accuracy_difference": -1.0,
         "left_stronger": 0}
        for judge in ("j1", "j2")
    ] + [
        {"bench": "b", "judge": "j1", "item": "easy", "model_left": "b", "model_right": "a",
         "epoch_left": 0, "epoch_right": 0, "log_left": "lb", "log_right": "la", "d_x": -1.0,
         "accuracy_left": 1.0, "accuracy_right": 0.0, "accuracy_difference": 1.0,
         "left_stronger": 1}
    ])

    direct = attach_direct_target(transcripts, accuracy, "Q2")
    pair = attach_pair_target(deduplicate_feature_pairs(pairs), accuracy, "Q2")

    assert direct.set_index("model").target_accuracy.to_dict() == {"a": 0.8, "b": 0.3}
    assert len(pair) == 1
    assert pair.loc[0, "left_stronger"] == 1
