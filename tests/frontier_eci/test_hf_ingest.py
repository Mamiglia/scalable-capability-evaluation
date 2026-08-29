from pathlib import Path
from runpy import run_path


compatible = run_path(Path(__file__).parents[2] / "scripts/frontier_eci/hf_ingest.py")["compatible"]


def test_only_exact_main_protocol_is_compatible():
    meta = {"require_both_correct": True, "include_reverse": True,
            "include_reasoning": False, "truncate_frac": None,
            "system_prompt": "assets/system_prompt/mcq.txt"}
    assert compatible(meta, "gpqa")
    assert not compatible(meta | {"include_reasoning": True}, "gpqa")
    assert not compatible(meta | {"system_prompt": "assets/system_prompt/default.txt"}, "gpqa")
