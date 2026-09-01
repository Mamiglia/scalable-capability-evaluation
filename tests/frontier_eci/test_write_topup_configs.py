from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest
import yaml


write_configs = run_path(
    Path(__file__).parents[2] / "scripts/frontier_eci/write_topup_configs.py"
)["write_configs"]


def manifest_frame():
    return pd.DataFrame([
        {"bench": "gpqa", "judge": "gpt-5.2", "target": "model-b", "opponent": "model-a", "item": "i1"},
        {"bench": "gpqa", "judge": "gpt-5.2", "target": "model-b", "opponent": "model-a", "item": "i2"},
        {"bench": "gpqa", "judge": "gpt-5.2", "target": "model-b", "opponent": "model-a", "item": "i3"},
    ])


def base_configs():
    return {
        ("gpqa", "gpt-5.2"): {
            "system_prompt": "assets/system_prompt/mcq.txt",
            "transcripts": {"model-a": "a.eval", "model-b": "b.eval"},
            "comparisons": [["openrouter/openai/gpt-5.2", "model-a", "model-b"]],
        }
    }


def test_each_config_has_one_pair_and_frozen_items(tmp_path):
    outputs = write_configs(manifest_frame(), base_configs(), tmp_path)

    assert len(outputs) == 1
    config = yaml.safe_load(outputs[0].read_text())
    assert config["comparisons"] == [["openrouter/openai/gpt-5.2", "model-a", "model-b"]]
    assert config["only_items"] == ["i1", "i2", "i3"]
    assert config["include_reasoning"] is False
    assert config["phase1_items"] == 8
    assert config["cache"] is False
    assert config["log_root"] == "logs/frontier-eci-canonical-anchors/gpqa"


def test_reasoning_ablation_can_override_rendering_and_log_root(tmp_path):
    outputs = write_configs(
        manifest_frame(), base_configs(), tmp_path,
        include_reasoning=True,
        log_root_template="logs/reasoning_ablation/cot_on/{bench}",
    )
    config = yaml.safe_load(outputs[0].read_text())
    assert config["include_reasoning"] is True
    assert config["log_root"] == "logs/reasoning_ablation/cot_on/gpqa"


def test_matched_pair_items_requires_both_transcript_orders():
    verdicts = pd.DataFrame([
        {"bench": "gpqa", "judge": "j", "item": "i1", "model_left": "a", "model_right": "b"},
        {"bench": "gpqa", "judge": "j", "item": "i1", "model_left": "b", "model_right": "a"},
        {"bench": "gpqa", "judge": "j", "item": "i2", "model_left": "a", "model_right": "b"},
        {"bench": "gpqa", "judge": "other", "item": "i1", "model_left": "a", "model_right": "b"},
    ])
    matched = run_path(
        Path(__file__).parents[2] / "scripts/frontier_eci/write_topup_configs.py"
    )["matched_pair_items"](verdicts)
    assert matched.to_dict("records") == [{
        "bench": "gpqa", "judge": "j", "target": "b", "opponent": "a", "item": "i1",
    }]


def test_reverse_manifest_rows_share_one_unordered_pair_config(tmp_path):
    manifest = manifest_frame().iloc[:2].copy()
    manifest.loc[1, ["target", "opponent"]] = ["model-a", "model-b"]

    outputs = write_configs(manifest, base_configs(), tmp_path)

    assert len(outputs) == 1
    assert yaml.safe_load(outputs[0].read_text())["only_items"] == ["i1", "i2"]


def test_emits_manifest_pair_not_in_original_comparisons(tmp_path):
    manifest = manifest_frame().iloc[:1].copy()
    manifest.loc[0, ["target", "opponent"]] = ["model-a", "model-c"]
    bases = base_configs()
    bases[("gpqa", "gpt-5.2")]["transcripts"]["model-c"] = "c.eval"

    config = yaml.safe_load(write_configs(manifest, bases, tmp_path)[0].read_text())

    assert config["comparisons"] == [["openrouter/openai/gpt-5.2", "model-a", "model-c"]]


def test_resolves_manifest_model_alias_to_transcript_key(tmp_path):
    manifest = manifest_frame().iloc[:1].copy()
    manifest.loc[0, ["target", "opponent"]] = ["model-a", "model-c"]
    bases = base_configs()
    bases[("gpqa", "gpt-5.2")]["transcripts"]["modelc"] = "c.eval"

    config = yaml.safe_load(write_configs(manifest, bases, tmp_path)[0].read_text())

    assert config["comparisons"] == [["openrouter/openai/gpt-5.2", "model-a", "modelc"]]


def test_missing_manifest_source_names_the_model_and_row_count(tmp_path):
    manifest = manifest_frame().iloc[:1].copy()
    manifest.loc[0, ["target", "opponent"]] = ["model-a", "not-imported"]

    with pytest.raises(ValueError, match=r"not-imported \(1 rows\)"):
        write_configs(manifest, base_configs(), tmp_path)


def test_extra_transcripts_complete_missing_manifest_pair(tmp_path):
    manifest = manifest_frame().iloc[:1].copy()
    manifest.loc[0, ["target", "opponent"]] = ["model-a", "model-c"]

    outputs = write_configs(
        manifest, base_configs(), tmp_path,
        extra_transcripts={"gpqa": {"model-c": "c.eval"}},
    )

    config = yaml.safe_load(outputs[0].read_text())
    assert len(outputs) == 1
    assert config["transcripts"]["model-c"] == "c.eval"
    assert config["comparisons"] == [["openrouter/openai/gpt-5.2", "model-a", "model-c"]]


def test_regeneration_removes_obsolete_pair_configs(tmp_path):
    bases = base_configs()
    bases[("gpqa", "gpt-5.2")]["transcripts"]["model-c"] = "c.eval"
    first = manifest_frame().iloc[:1].copy()
    second = first.copy()
    second.loc[0, ["target", "opponent"]] = ["model-a", "model-c"]

    stale = write_configs(first, bases, tmp_path)[0]
    current = write_configs(second, bases, tmp_path)[0]

    assert not stale.exists()
    assert current.exists()


def test_failed_regeneration_preserves_existing_configs(tmp_path):
    existing = write_configs(manifest_frame(), base_configs(), tmp_path)[0]
    before = existing.read_text()
    invalid = manifest_frame().iloc[:1].copy()
    invalid.loc[0, ["target", "opponent"]] = ["model-a", "not-imported"]

    with pytest.raises(ValueError, match="not-imported"):
        write_configs(invalid, base_configs(), tmp_path)

    assert existing.read_text() == before
