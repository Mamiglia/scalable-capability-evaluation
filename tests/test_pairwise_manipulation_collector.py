import importlib.util
from pathlib import Path
from types import SimpleNamespace

import yaml


SCRIPT = (Path(__file__).parents[1] / "scripts" / "pairwise-newjudges" / "analyze.py")
SPEC = importlib.util.spec_from_file_location("pairwise_analyze", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_style_config_preserves_the_fixed_1500_cell_campaign():
    configs = SCRIPT.parent / "configs" / "manipulations"
    fixed = yaml.safe_load((configs / "trunc_first_10.yaml").read_text())
    style = yaml.safe_load((configs / "style_standardised_gpt.yaml").read_text())

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
