import json

import pytest

from model import sessions


def test_register_drop_and_unregister():
    sessions.register("key", "pod1", "http://pod1/v1")
    assert sessions.drop("key") == 0
    assert sessions.unregister("key", only_if_unused=True)
    assert not sessions.unregister("key")


def test_unregister_keeps_pod_that_was_re_taken():
    sessions.register("key", "pod1", "http://pod1/v1")
    with sessions.locked_state() as state:
        entry, is_creating = sessions.claim_or_join(state, "key", ready_timeout=60)
    assert entry["pod_id"] == "pod1" and not is_creating

    # Two users now: a keep-alive timer must not terminate the pod under them.
    assert not sessions.unregister("key", only_if_unused=True)
    assert sessions.drop("key") == 1


def test_claim_creation_then_others_wait():
    with sessions.locked_state() as state:
        entry, is_creating = sessions.claim_or_join(state, "key", ready_timeout=60)
    assert entry is None and not is_creating  # we claimed creation

    with sessions.locked_state() as state:
        entry, is_creating = sessions.claim_or_join(state, "key", ready_timeout=60)
    assert entry is None and is_creating  # someone else is on it


def test_stale_creation_claim_is_taken_over(monkeypatch):
    with sessions.locked_state() as state:
        state["key"] = {"status": "creating", "created_at": 0, "pid": None}
        entry, is_creating = sessions.claim_or_join(state, "key", ready_timeout=60)
    assert entry is None and not is_creating


def test_dead_creator_claim_is_taken_over(monkeypatch):
    import time

    monkeypatch.setattr(sessions.os, "kill", lambda pid, sig: (_ for _ in ()).throw(OSError))
    with sessions.locked_state() as state:
        state["key"] = {"status": "creating", "created_at": time.time(), "pid": 424242}
        entry, is_creating = sessions.claim_or_join(state, "key", ready_timeout=60)
    assert entry is None and not is_creating


def test_corrupt_state_file_raises():
    sessions.STATE_FILE.write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable"), sessions.locked_state():
        pass


def test_state_survives_round_trip():
    sessions.register("key", "pod1", "http://pod1/v1")
    assert json.loads(sessions.STATE_FILE.read_text())["key"]["pod_id"] == "pod1"
