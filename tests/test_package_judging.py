import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import package_judging
import upload_judging


def test_package_identity_topups_and_current_prompts(tmp_path, monkeypatch):
    canonical = {"llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct"}
    upper = ("Llama-3.1-8B-Instruct", "pair", "1")
    lower = ("llama-3.1-8b-instruct", "pair", "1")
    assert package_judging.canonical_key(upper, canonical) == package_judging.canonical_key(
        lower, canonical
    )
    assert package_judging.selected_keys(
        [("judge", "pair", "0"), ("judge", "extra", "0"), ("judge", "pair", "old")],
        {"pair"},
        {"0", "1"},
    ) == [("judge", "pair", "0")]

    topups = []
    for root_name, filename in (("old", "a.eval"), ("new", "b.eval")):
        path = tmp_path / root_name / "epoch-0/judge/pair/topup" / filename
        path.parent.mkdir(parents=True)
        path.touch()
        topups.append(path)
    monkeypatch.setattr(
        package_judging,
        "read_eval_log",
        lambda *_args, **_kwargs: SimpleNamespace(status="success"),
    )
    assert package_judging.topup_logs([tmp_path / "old", tmp_path / "new"]) == {
        ("judge", "pair", "0"): topups
    }

    current = SimpleNamespace(id="item_1", input="Input question:\n---\ncurrent\n---\n")
    removed = SimpleNamespace(id="removed_1", input="old")
    assert package_judging.current_samples(
        [current, removed], {"item_1": "Input question:\n---\n\ncurrent\n\n---\n"}
    ) == [current]
    with pytest.raises(AssertionError):
        package_judging.current_samples(
            [SimpleNamespace(id="item_1", input="changed")], {"item_1": current.input}
        )

    older, newer = tmp_path / "older.eval", tmp_path / "newer.eval"
    older.touch()
    newer.touch()

    def logs(path, header_only=False):
        name = Path(path).name
        if header_only:
            return SimpleNamespace(
                status="success",
                eval=SimpleNamespace(created={"older.eval": "1", "newer.eval": "2"}[name]),
            )
        answer = "1" if name == "older.eval" else "2"
        return SimpleNamespace(
            samples=[
                SimpleNamespace(
                    id="item_1",
                    epoch=1,
                    scores={"judge_scorer": SimpleNamespace(answer=answer)},
                )
            ]
        )

    monkeypatch.setattr(package_judging, "read_eval_log", logs)
    assert package_judging.latest_success_path([newer, older]) == newer
    base = SimpleNamespace(
        id="item_1", epoch=1, scores={"judge_scorer": SimpleNamespace(answer=None)}
    )
    assert package_judging.overlay_answered([base], [older, newer])[0].scores[
        "judge_scorer"
    ].answer == "2"

    rescored = object()
    monkeypatch.setattr(package_judging, "inspect_score", lambda *_args, **_kwargs: rescored)
    assert package_judging.rescore_packaged(SimpleNamespace()) is rescored


def test_upload_preserves_live_layout_and_is_additions_only(tmp_path, monkeypatch):
    aime = Path("aime/aime2024/o4-mini/pair/new.eval")
    assert upload_judging.repo_path(
        aime, {Path("aime/aime2024"): "b956d0d4"}
    ) == Path("aime/aime2024/o4-mini/b956d0d4/new.eval")
    gpqa = Path("pilot/gpqa/Llama-3.1-8B-Instruct/pair/new.eval")
    assert upload_judging.repo_path(
        gpqa,
        {},
        {(Path("pilot/gpqa"), "llama-3.1-8b-instruct"): "llama-3.1-8b-instruct"},
    ) == Path("pilot/gpqa/llama-3.1-8b-instruct/pair/new.eval")

    mirror = tmp_path / "mirror"
    old = mirror / "pilot/gpqa/judge/pair/old.eval"
    old.parent.mkdir(parents=True)
    old.touch()
    assert upload_judging.supersede_operations([old], mirror, enabled=False) == []
    assert len(upload_judging.supersede_operations([old], mirror, enabled=True)) == 2

    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_judging", str(tmp_path / "packaged"), "--mirror", str(mirror), "--commit"],
    )
    with pytest.raises(SystemExit):
        upload_judging.main()
