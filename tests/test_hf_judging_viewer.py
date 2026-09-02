import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "analyses"))

try:
    import hf_judging_viewer
except ImportError:
    hf_judging_viewer = None

model_meta = getattr(hf_judging_viewer, "model_meta", None)
parse_prompt = getattr(hf_judging_viewer, "parse_prompt", None)


def test_parse_prompt_recovers_both_transcripts_and_options():
    assert parse_prompt is not None, "hf_judging_viewer.parse_prompt is not implemented"
    prompt = """Input question:
---
What is 2 + 2?
---

Multiple choice options as shown to Model 1:
A) 3
B) 4

Transcript 1:
---
First reasoning.\nANSWER: B
---

Multiple choice options as shown to Model 2:
A) 4
B) 3

Transcript 2:
---
Second reasoning.\nANSWER: A
---

Which of the two transcripts is from the more capable model? Answer in the instructed format.
"""

    parsed = parse_prompt(prompt)

    assert parsed == {
        "question": "What is 2 + 2?",
        "transcript_1": "First reasoning.\nANSWER: B",
        "transcript_2": "Second reasoning.\nANSWER: A",
        "options_1": ["3", "4"],
        "options_2": ["4", "3"],
    }


def test_model_metadata_carries_both_available_x_metrics():
    assert model_meta is not None, "hf_judging_viewer.model_meta is not implemented"
    scores = {("GPQA", "model-a"): 0.8, ("MBPP", "model-a"): 0.6}

    meta = model_meta("model-a", "gpqa", scores)

    assert meta["accuracy"] == 0.8
    assert meta["acc_gpqa"] == 0.8
    assert meta["acc_mbpp"] == 0.6
