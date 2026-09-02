import importlib.util
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


SCRIPT = (Path(__file__).parents[1] / "scripts" / "pairwise-newjudges" / "analyze.py")
SPEC = importlib.util.spec_from_file_location("pairwise_analyze", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def campaign_config_and_rows():
    comparisons = [(f"major-{index}", f"minor-{index}") for index in range(50)]
    sample_ids = {
        f"{major}-vs-{minor}": [f"item-{item}" for item in range(30)]
        for major, minor in comparisons
    }
    config = {
        "condition": "style_standardised_gpt",
        "comparisons": comparisons,
        "sample_ids": sample_ids,
    }
    rows = []
    for judge in ("qwen3-8b", "qwen3-32b", "gpt-oss-20b"):
        for major, minor in comparisons:
            for item in sample_ids[f"{major}-vs-{minor}"]:
                for order in ("1", "2"):
                    rows.append({key: "" for key in analysis.FIELDS} | {
                        "bench": "gpqa", "condition": config["condition"],
                        "judge": judge, "major": major, "minor": minor,
                        "item_id": item, "order": order, "parse_ok": False,
                    })
    return config, rows


def test_style_config_preserves_the_fixed_1500_cell_campaign():
    configs = SCRIPT.parent / "configs" / "manipulations"
    fixed = yaml.safe_load((configs / "trunc_first_10.yaml").read_text())
    style = yaml.safe_load((configs / "style_standardised_gpt.yaml").read_text())

    assert analysis.MANIPULATION_LOG_ROOT == (
        SCRIPT.parents[2] / "logs/pairwise-newjudges/manipulations"
    )
    assert analysis.CAMPAIGN_JUDGES == (
        "qwen3-8b", "qwen3-32b", "gpt-oss-20b",
    )
    assert analysis.STYLE_REWRITE_VERDICTS == (
        SCRIPT.parents[2]
        / "judge_results/robustness/style_standardised_gpt/verdicts_rewrite.csv"
    )
    assert style["comparisons"] == fixed["comparisons"]
    assert style["sample_ids"] == fixed["sample_ids"]
    assert len(style["sample_ids"]) == 50
    assert {len(items) for items in style["sample_ids"].values()} == {30}
    assert sum(map(len, style["sample_ids"].values())) == 1500


def test_collect_manipulation_uses_condition_judge_pair_layout(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    pair = logs / "style_standardised_gpt" / "qwen3-8b" / "a-vs-b"
    pair.mkdir(parents=True)
    (pair / "success.eval").touch()
    old = logs / "paraphrase_haiku" / "qwen3-8b" / "a-vs-b"
    old.mkdir(parents=True)
    (old / "success.eval").touch()

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "style_standardised_gpt.yaml").write_text(yaml.safe_dump({
        "condition": "style_standardised_gpt",
        "transcripts": {"a": "a.eval", "b": "b.eval"},
        "include_reasoning": False,
        "max_length": 100,
    }))
    score = SimpleNamespace(answer="1", metadata={"confidence": 4})
    log = SimpleNamespace(
        status="success",
        samples=[SimpleNamespace(id="item_1", scores={"judge": score})],
    )
    monkeypatch.setattr(analysis, "read_eval_log", lambda *_args, **_kwargs: log)
    monkeypatch.setattr(analysis, "_transcript_lengths", lambda *_args, **_kwargs: {"item": 5})

    rows = analysis.collect_manipulations(
        logs, configs, ["style_standardised_gpt"], {"qwen3-8b"}
    )

    assert [{key: row[key] for key in (
        "bench", "condition", "judge", "major", "minor", "item_id", "order",
        "picked", "picked_major", "confidence", "parse_ok",
    )} for row in rows] == [{
        "bench": "gpqa", "condition": "style_standardised_gpt",
        "judge": "qwen3-8b", "major": "a", "minor": "b",
        "item_id": "item", "order": "1", "picked": "1",
        "picked_major": True, "confidence": 4, "parse_ok": True,
    }]


def test_manipulation_cli_uses_campaign_defaults(monkeypatch):
    captured = {}

    def collect(root, configs, conditions, judges):
        captured.update(root=root, configs=configs, conditions=conditions, judges=judges)
        return []

    def write(rows, config, out):
        captured.update(rows=rows, config=config, out=out)

    monkeypatch.setattr(analysis, "collect_manipulations", collect)
    monkeypatch.setattr(analysis, "write_manipulation_verdicts", write)
    monkeypatch.setattr(
        analysis.sys, "argv", ["analyze.py", "--conditions", "style_standardised_gpt"]
    )

    analysis.main()

    assert captured["root"] == analysis.MANIPULATION_LOG_ROOT
    assert captured["conditions"] == ["style_standardised_gpt"]
    assert captured["judges"] == set(analysis.CAMPAIGN_JUDGES)
    assert captured["out"] == analysis.STYLE_REWRITE_VERDICTS


def test_incomplete_or_duplicate_campaign_refuses_to_touch_existing_output(tmp_path):
    config, rows = campaign_config_and_rows()
    out = tmp_path / "verdicts_rewrite.csv"
    out.write_text("preserve me")

    with pytest.raises(ValueError, match="9,000 unique rows"):
        analysis.write_manipulation_verdicts(rows[:-1], config, out)
    assert out.read_text() == "preserve me"

    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="9,000 unique rows"):
        analysis.write_manipulation_verdicts(rows, config, out)
    assert out.read_text() == "preserve me"


def test_complete_campaign_is_written_atomically_with_unparsed_rows(tmp_path):
    config, rows = campaign_config_and_rows()
    out = tmp_path / "verdicts_rewrite.csv"

    analysis.write_manipulation_verdicts(rows, config, out)

    with out.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 9000
    assert written[0]["parse_ok"] == "False"
    assert set(tmp_path.iterdir()) == {out}
