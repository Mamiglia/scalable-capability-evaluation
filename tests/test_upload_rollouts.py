import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "collect"))

import upload_rollouts


def test_rerun_uploads_require_explicit_opt_in_and_commits_are_revision_guarded(
    monkeypatch, tmp_path
):
    final = Path("model/0/final.eval")
    partial = Path("model/0/reruns/partial.eval")

    assert upload_rollouts.upload_candidate(final, "source.eval", include_reruns=False)
    assert not upload_rollouts.upload_candidate(partial, None, include_reruns=False)
    assert upload_rollouts.upload_candidate(partial, None, include_reruns=True)

    collection = tmp_path / "gpqa" / "hash"
    log = collection / "model" / "0" / "final.eval"
    log.parent.mkdir(parents=True)
    log.touch()
    calls = []

    class Api:
        def dataset_info(self, repo_id):
            assert repo_id == upload_rollouts.REPO
            return SimpleNamespace(sha="parent-sha")

        def list_repo_files(self, *_args, revision, **_kwargs):
            assert revision == "parent-sha"
            return []

        def create_commit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="commit-sha")

    monkeypatch.setattr(upload_rollouts, "HfApi", Api)
    monkeypatch.setattr(
        upload_rollouts,
        "read_eval_log",
        lambda *_args, **_kwargs: SimpleNamespace(
            eval=SimpleNamespace(metadata={"finalised_from": "source.eval"})
        ),
    )
    monkeypatch.setattr(sys, "argv", ["upload_rollouts", str(collection), "--commit"])

    assert upload_rollouts.main() == 0
    assert calls[0]["parent_commit"] == "parent-sha"
