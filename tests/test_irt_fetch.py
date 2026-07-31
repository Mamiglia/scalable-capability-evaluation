"""
Unit tests for the IRT fetch downloader.
"""

import io
from pathlib import Path

import pandas as pd
import pytest

import irt.fetch as fetch
from irt.fetch import (
    FetchConfig,
    LOGS_COLUMN,
    MODEL_COLUMN,
    cdn_url,
    download,
    main,
    select_models,
)


def make_frame(rows):
    """Build a leaderboard-shaped frame from (name, url) pairs."""
    return pd.DataFrame([{MODEL_COLUMN: name, LOGS_COLUMN: url} for name, url in rows])


class FakeUrlopen:
    """Stand-in for urllib.request.urlopen that records the URLs it is asked for."""

    def __init__(self, payload: bytes = b"log-bytes"):
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        return io.BytesIO(self.payload)


# --- select_models ---

def test_select_models_sorts_into_buckets():
    frame = make_frame(
        [
            ("has-log", "https://s3/has-log.eval"),
            ("no-log", ""),  # present in the CSV but no URL
        ]
    )
    resolved, missing_url, not_found = select_models(
        frame, ["has-log", "no-log", "absent"]
    )
    assert resolved == [("has-log", "https://s3/has-log.eval")]
    assert missing_url == ["no-log"]
    assert not_found == ["absent"]


def test_select_models_all_selects_every_url():
    frame = make_frame(
        [
            ("a", "https://s3/a.eval"),
            ("b", "   "),  # whitespace-only → treated as no URL after strip
            ("c", "https://s3/c.eval"),
        ]
    )
    for models in ("all", None):
        resolved, missing_url, not_found = select_models(frame, models)
        assert resolved == [("a", "https://s3/a.eval"), ("c", "https://s3/c.eval")]
        assert missing_url == ["b"]
        assert not_found == []


# --- download ---

def test_download_writes_file_and_removes_tmp(tmp_path, monkeypatch):
    fake = FakeUrlopen(b"payload")
    monkeypatch.setattr(fetch, "urlopen", fake)

    created = download("model-a", "https://s3/a.eval", tmp_path)

    assert created is True
    assert (tmp_path / "model-a.eval").read_bytes() == b"payload"
    assert not (tmp_path / "model-a.eval.tmp").exists()
    assert fake.calls[0].full_url == cdn_url("https://s3/a.eval")


def test_download_skips_existing_and_does_not_redownload(tmp_path, monkeypatch):
    (tmp_path / "model-a.eval").write_bytes(b"already here")
    fake = FakeUrlopen(b"new bytes")
    monkeypatch.setattr(fetch, "urlopen", fake)

    created = download("model-a", "https://s3/a.eval", tmp_path)

    assert created is False
    assert fake.calls == []  # never touched the network
    assert (tmp_path / "model-a.eval").read_bytes() == b"already here"  # left intact


def test_download_cleans_tmp_on_failure(tmp_path, monkeypatch):
    class BrokenStream(io.BytesIO):
        def read(self, *args, **kwargs):
            raise OSError("truncated read")

    monkeypatch.setattr(fetch, "urlopen", lambda url, timeout=None: BrokenStream(b"x"))

    with pytest.raises(OSError):
        download("model-a", "https://s3/a.eval", tmp_path)

    # the partial .tmp is removed and no final file is left behind
    assert not (tmp_path / "model-a.eval.tmp").exists()
    assert not (tmp_path / "model-a.eval").exists()


def test_download_sends_waf_token_as_cookie(tmp_path, monkeypatch):
    fake = FakeUrlopen(b"payload")
    monkeypatch.setattr(fetch, "urlopen", fake)

    download("model-a", "https://s3/a.eval", tmp_path, token="tok123")

    request = fake.calls[0]
    assert request.full_url == cdn_url("https://s3/a.eval")  # host-swapped to the CDN
    assert request.headers.get("Cookie") == "aws-waf-token=tok123"


# --- main ---

def test_main_downloads_missing_only_and_is_idempotent(tmp_path, monkeypatch):
    csv = tmp_path / "leaderboard.csv"
    make_frame(
        [
            ("a", "https://s3/a.eval"),
            ("b", "https://s3/b.eval"),
            ("top", ""),  # no URL → reported, never downloaded
        ]
    ).to_csv(csv, index=False)
    out_dir = tmp_path / "out"

    fake = FakeUrlopen()
    monkeypatch.setattr(fetch, "urlopen", fake)
    cfg = FetchConfig(csv=str(csv), out_dir=str(out_dir), models="all")

    assert main(cfg) == 0
    assert {p.name for p in out_dir.glob("*.eval")} == {"a.eval", "b.eval"}
    assert len(fake.calls) == 2

    # second run: everything already present → nothing re-downloaded
    fake.calls.clear()
    assert main(cfg) == 0
    assert fake.calls == []


# --- model names/logs matches the CSV ---

EPOCH_CSV = Path(__file__).resolve().parents[1] / "external_data/epoch-gpqa/gpqa_diamond.csv"


@pytest.mark.skipif(not EPOCH_CSV.exists(), reason="Epoch leaderboard CSV not present")
def test_column_constants_match_epoch_csv():
    columns = list(pd.read_csv(EPOCH_CSV, nrows=0).columns)
    assert MODEL_COLUMN in columns, f"{MODEL_COLUMN!r} not in CSV headers: {columns}"
    assert LOGS_COLUMN in columns, f"{LOGS_COLUMN!r} not in CSV headers: {columns}"
