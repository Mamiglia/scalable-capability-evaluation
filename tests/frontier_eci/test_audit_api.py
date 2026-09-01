from pathlib import Path
from runpy import run_path

import pandas as pd
import pytest


MODULE = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/audit_api.py")


def test_missing_provider_is_rejected_even_when_usage_is_zero():
    calls = pd.DataFrame([{
        "judge": "gpt-5.2", "provider": None, "cost": 0.0,
        "input_tokens": 0, "output_tokens": 0,
    }])
    with pytest.raises(ValueError, match="responses lack provider provenance"):
        MODULE["validate_provider_provenance"](calls)
