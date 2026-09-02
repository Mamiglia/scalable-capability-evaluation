import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "pairwise-newjudges"))

from run import validated_truncate_keep


def test_truncate_keep_defaults_to_first():
    assert validated_truncate_keep({}) == "first"


def test_truncate_keep_accepts_last():
    assert validated_truncate_keep({"truncate_keep": "last"}) == "last"


def test_truncate_keep_rejects_bad_value():
    with pytest.raises(ValueError, match="truncate_keep"):
        validated_truncate_keep({"truncate_keep": "Last"})
