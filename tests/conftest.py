import pytest


@pytest.fixture(autouse=True)
def isolated_runpod_state(tmp_path, monkeypatch):
    """Keep tests off the real /tmp session file, which a live collection may be using."""
    from model import sessions

    monkeypatch.setattr(sessions, "STATE_FILE", tmp_path / "runpod_sessions.json")
    monkeypatch.setattr(sessions, "LOCK_FILE", tmp_path / "runpod_sessions.lock")


@pytest.fixture(autouse=True)
def clear_transcript_cache():
    from judge import dataset

    dataset._STREAMED.clear()
    dataset._RANDOM.clear()
    dataset.read_header.cache_clear()


@pytest.fixture(autouse=True)
def no_runpod_api(monkeypatch):
    """Placement lookups hit Runpod's API; tests never should."""
    from model import runpod

    monkeypatch.setattr(runpod, "placement", lambda pod_id: {"machine": None, "datacenter": None})
