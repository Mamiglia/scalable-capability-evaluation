import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import run_judging
import gen_pairs_judge_config


def test_exact_topup_runs_only_missing_orders_with_model_yaml_args(monkeypatch, tmp_path):
    config = tmp_path / "topup.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "log_root": str(tmp_path / "logs"),
                "transcripts": {"major": "major.eval", "minor": "minor.eval"},
                "comparisons": [["openrouter/judge", "major", "minor"]],
                "only_sample_ids": {"major-vs-minor": ["item_2"]},
                "generate_args": {"max_tokens": 123},
                "max_connections": 7,
            }
        )
    )
    task = SimpleNamespace(dataset=[SimpleNamespace(id="item_1"), SimpleNamespace(id="item_2")])
    calls = []
    monkeypatch.setattr(run_judging, "judge_task", lambda **_kwargs: task)
    monkeypatch.setattr(run_judging, "judge_config", lambda slug: slug)

    class FakeModel:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def eval_kwargs(self):
            return {
                "model": "openrouter/judge",
                "model_args": {"provider": {"only": ["p"]}},
            }

    monkeypatch.setattr(run_judging, "Model", FakeModel)
    monkeypatch.setattr(
        run_judging,
        "inspect_eval",
        lambda **kwargs: calls.append(kwargs) or [SimpleNamespace(status="success")],
    )
    monkeypatch.setattr(
        run_judging,
        "eval_set",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("eval_set used")),
    )
    monkeypatch.setattr(sys, "argv", ["run_judging", "--config", str(config)])

    assert run_judging.main() == 0
    assert calls[0]["sample_id"] == ["item_2"]
    assert calls[0]["model_args"] == {"provider": {"only": ["p"]}}
    assert calls[0]["max_tokens"] == 123
    assert calls[0]["max_connections"] == 7

    assert gen_pairs_judge_config.judge_overrides(
        {
            "generate_args": {"max_tokens": 99},
            "max_connections": 5,
        },
        ["openai-api/judge"],
        "openai-api/judge",
        8,
    ) == {"generate_args": {"max_tokens": 99}, "max_connections": 5}
    assert gen_pairs_judge_config.judge_overrides(
        {}, ["openai-api/judge"], "openai-api/judge", 8
    ) == {"max_connections": 8}
