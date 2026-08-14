import sys
from pathlib import Path

import pytest

# Add src directory to python path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture(autouse=True)
def isolated_runpod_state(tmp_path, monkeypatch):
    """Keep tests off the real /tmp session file, which a live collection may be using."""
    from model import sessions

    monkeypatch.setattr(sessions, "STATE_FILE", tmp_path / "runpod_sessions.json")
    monkeypatch.setattr(sessions, "LOCK_FILE", tmp_path / "runpod_sessions.lock")
