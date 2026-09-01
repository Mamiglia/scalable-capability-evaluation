import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "collect"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import final_transcripts
import finalize_rollout_logs


def test_final_selection_ignores_partial_and_failed_final_logs(tmp_path, monkeypatch):
    epoch = tmp_path / "0"
    epoch.mkdir()
    paths = {
        name: epoch / name
        for name in (
            "full.eval",
            "partial.eval",
            "limited.eval",
            "good-final.eval",
            "bad-final.eval",
        )
    }
    for path in paths.values():
        path.touch()

    def header(path, header_only=True):
        name = Path(path).name
        return SimpleNamespace(
            status="error" if name == "bad-final.eval" else "success",
            eval=SimpleNamespace(
                created={
                    "full.eval": "1",
                    "partial.eval": "4",
                    "limited.eval": "5",
                    "good-final.eval": "2",
                    "bad-final.eval": "3",
                }[name],
                config=SimpleNamespace(
                    sample_id=["x"] if name == "partial.eval" else None,
                    limit=1 if name == "limited.eval" else None,
                ),
                metadata={"finalised_from": "full.eval"} if "final" in name else {},
            ),
            results=SimpleNamespace(completed_samples=30),
        )

    monkeypatch.setattr(final_transcripts, "read_eval_log", header)
    monkeypatch.setattr(finalize_rollout_logs, "read_eval_log", header)

    assert finalize_rollout_logs.log_kind(paths["partial.eval"]) == "rerun"
    assert finalize_rollout_logs.log_kind(paths["limited.eval"]) == "limited"
    assert final_transcripts.pick(epoch, paths["full.eval"]) == paths["good-final.eval"]

    paths["good-final.eval"].unlink()
    paths["bad-final.eval"].unlink()
    assert final_transcripts.pick(epoch, paths["partial.eval"]) == paths["full.eval"]

    output = tmp_path / "stage.eval"
    output.touch()
    monkeypatch.setattr(
        finalize_rollout_logs.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="failed", stderr=""),
    )
    with pytest.raises(RuntimeError):
        finalize_rollout_logs.run(["rescore"], output)
