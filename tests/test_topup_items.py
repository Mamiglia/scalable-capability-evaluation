import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import topup_items


def test_matching_sample_ids_preserves_order_level_gaps():
    logged = "Input question:\n---\nquestion\n---\n\nTranscript 1:\n---\nanswer\n---\n"
    current = "Input question:\n---\n\nquestion\n\n---\n\nTranscript 1:\n---\n\nanswer\n\n---\n"
    expected = {"item_1": current, "item_2": current}

    assert topup_items.matching_sample_ids(
        [SimpleNamespace(id="item_1", input=logged), SimpleNamespace(id="item_2", input=logged)],
        expected,
    ) == {"item_1", "item_2"}
    assert topup_items.matching_sample_ids(
        [
            SimpleNamespace(id="item_1", input=logged),
            SimpleNamespace(id="item_2", input=logged.replace("answer", "stale")),
        ],
        expected,
    ) == {"item_1"}
