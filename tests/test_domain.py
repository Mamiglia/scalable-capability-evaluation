from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from domain.cli import difficulties, guess_rate
from ranking import verdicts


def test_guess_rate():
    assert guess_rate(SimpleNamespace(choices=["a", "b", "c", "d"], input="q")) == 0.25
    hle = "Which?\n\nAnswer Choices:\nA. x\nB. y\nC. z\nD. w\nE. v"
    assert guess_rate(SimpleNamespace(choices=None, input=hle)) == 0.2
    assert guess_rate(SimpleNamespace(choices=None, input="What is 2+2?")) == 0.0


def test_difficulties_quartiles_follow_correctness():
    # 12 items: item i is answered correctly by the (4 - i // 3) strongest of 4 models.
    rows = [dict(bench="b", model=f"m{m}", item=str(i), repeat=0, correct=m < 4 - i // 3, guess=0.0)
            for i in range(12) for m in range(4)]
    items = difficulties(pd.DataFrame(rows), warmup=100, samples=100)
    assert sorted(items.quartile.value_counts()) == [3, 3, 3, 3]
    by_item = items.set_index("item")
    assert by_item.loc[["0", "1", "2"], "quartile"].tolist() == [1, 1, 1]
    assert by_item.loc[["9", "10", "11"], "quartile"].tolist() == [4, 4, 4]
    assert np.all(np.diff(by_item.loc[[str(i) for i in range(0, 12, 3)], "b"]) > 0)


def test_verdicts_from_logs(monkeypatch, tmp_path):
    (tmp_path / "x.eval").touch()
    header = SimpleNamespace(status="success", eval=SimpleNamespace(
        model="judge", metadata={"bench": "gpqa", "major": "big", "minor": "small"}))
    summaries = [
        SimpleNamespace(id="q1_1", scores={"s": SimpleNamespace(answer="1")}),  # big shown left, picked
        SimpleNamespace(id="q1_2", scores={"s": SimpleNamespace(answer="1")}),  # small shown left, picked
        SimpleNamespace(id="q2_1", scores={"s": SimpleNamespace(answer=None)}),  # no verdict
    ]
    monkeypatch.setattr(verdicts, "read_eval_log", lambda *a, **k: header)
    monkeypatch.setattr(verdicts, "read_eval_log_sample_summaries", lambda *a, **k: summaries)
    table = verdicts.from_logs(tmp_path)
    assert table.to_dict("records") == [
        dict(bench="gpqa", judge="judge", item="q1", model_left="big", model_right="small", pick_left=1.0),
        dict(bench="gpqa", judge="judge", item="q1", model_left="small", model_right="big", pick_left=1.0),
    ]
