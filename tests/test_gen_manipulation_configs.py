import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "pairwise-newjudges"))

from gen_manipulation_configs import (
    COVERAGE_JUDGES,
    ITEMS_PER_PAIR,
    TRUNCATION_CONDITIONS,
    covered_items,
    filtered_comparisons,
    sample_pair_items,
)


def test_100_percent_is_the_existing_untruncated_baseline():
    assert {frac for _, _, frac in TRUNCATION_CONDITIONS} == {0.1, 0.5, 0.9}


def _row(item_id, judge, order, parse_ok=True):
    return dict(major="m1", minor="m2", item_id=item_id, judge=judge, order=order, parse_ok=parse_ok)


def test_covered_items_drops_item_with_unparsed_row():
    rows = []
    for judge in COVERAGE_JUDGES:
        for order in (1, 2):
            rows.append(_row("item-covered", judge, order))
    for judge in COVERAGE_JUDGES:
        for order in (1, 2):
            # one cell failed to parse - the item must not count as covered
            bad = judge == COVERAGE_JUDGES[0] and order == 1
            rows.append(_row("item-unparsed", judge, order, parse_ok=not bad))
    verdicts = pd.DataFrame(rows)

    assert covered_items(verdicts, "m1", "m2") == ["item-covered"]


def test_sample_pair_items_is_deterministic():
    covered = [f"item-{i}" for i in range(ITEMS_PER_PAIR * 2)]
    assert sample_pair_items("m1", "m2", covered) == sample_pair_items("m1", "m2", covered)
    assert len(sample_pair_items("m1", "m2", covered)) == ITEMS_PER_PAIR


def test_filtered_comparisons_drops_either_side_case_insensitively():
    comparisons = [["DeepSeek-R1", "phi-4"], ["glm-5", "deepseek-chat"], ["glm-5", "phi-4"]]
    assert filtered_comparisons(comparisons, r"deepseek") == [["glm-5", "phi-4"]]


def test_filtered_comparisons_is_a_noop_without_a_pattern():
    comparisons = [["DeepSeek-R1", "phi-4"]]
    assert filtered_comparisons(comparisons, None) == comparisons
