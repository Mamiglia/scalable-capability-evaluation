import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit_surprisal_cache import match_legacy_rows, migrate_reusable


def test_match_legacy_rows_uses_first_cached_content_and_exact_text():
    keys = {"bench": "gpqa", "model": "m", "item": "q", "epoch": 0, "log": "run"}
    legacy = pd.DataFrame([
        keys | {"question": "Q", "transcript": "first"},
        keys | {"question": "Q", "transcript": "later"},
    ])
    frozen = pd.DataFrame([
        keys | {"sample_epoch": 1, "content_hash": "a", "question": "Q", "transcript": "first"},
        keys | {"sample_epoch": 2, "content_hash": "b", "question": "Q", "transcript": "later"},
    ])

    matched = match_legacy_rows(legacy, frozen)
    assert matched.status.tolist() == ["exact", "text_mismatch"]


def test_migrate_reusable_writes_content_addressed_copy(tmp_path):
    source = tmp_path / "legacy.npz"
    np.savez(source, full=[1.0], noq=[2.0], local=[3.0], offsets=[[0, 1]])
    audit = pd.DataFrame([{
        "bench": "gpqa", "model": "m", "item": "q", "epoch": 0, "log": "run",
        "sample_epoch": 1, "content_hash": "a" * 64, "scorer": "llama",
        "legacy_cache": str(source), "status": "reusable",
    }])

    migrated = migrate_reusable(audit, tmp_path / "new")
    target = Path(migrated.iloc[0].new_cache)
    assert source.exists() and target.exists()
    assert np.load(target)["content_hash"].item() == "a" * 64
