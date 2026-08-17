"""HF snapshot rows preserve the judge's left/right orientation."""

from types import SimpleNamespace

import pytest

try:
    from prepare_hf import row_from_sample
except ImportError:
    row_from_sample = None


def test_row_uses_benchmark_capability_and_prompt_position():
    assert row_from_sample is not None, "prepare_hf.row_from_sample is not implemented"
    prompt = """Input question:
---
Question?
---

Transcript 1:
---
short
---

Transcript 2:
---
the much longer answer
---

Which of the two transcripts is from the more capable model?
"""
    sample = SimpleNamespace(
        id="item-7_2",
        input=prompt,
        target="2",
        scores={"judge_scorer": SimpleNamespace(answer="1")},
    )
    metadata = {
        "major_path": "data/gpqa/hash/strong/0/run.eval",
        "minor_path": "data/gpqa/hash/weak/0/run.eval",
    }

    row = row_from_sample(
        sample, metadata, "judge-a", "gpqa",
        {("GPQA", "strong"): 0.8, ("GPQA", "weak"): 0.3},
    )

    assert row == {
        "chosen_left": 1,
        "judge": "judge-a",
        "item": "gpqa:item-7",
        "transcript_condition": "gpqa",
        "bench": "gpqa",
        "model_pair": "strong|weak",
        "left_model": "weak",
        "right_model": "strong",
        "left_chars": 5,
        "right_chars": 22,
        "left_capability": 0.3,
        "right_capability": 0.8,
        "length_difference": pytest.approx(-1.3437347467),
        "capability_difference": -0.5,
        "same_model_family": 0,
        "self_comparison": 0,
    }
