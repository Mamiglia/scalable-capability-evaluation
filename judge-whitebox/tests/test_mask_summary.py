import numpy as np
import pandas as pd

import mask_summary


def toy_scores():
    """Two items x one pair x both orders; transcripts swap slots between orders.

    Content effects: transcript A carries +2 total, B carries -1, everywhere.
    A position bonus of +0.5 lands on whichever transcript is presented first.
    """
    rows = []
    for item in ("i0", "i1"):
        for order in (0, 1):
            first, second = ("A", "B") if order == 0 else ("B", "A")
            content = {"A": 2.0, "B": -1.0}
            rows.append({"item": item, "pair_uid": f"{item}p", "order": order,
                         "group": "transcript_1", "score": content[first] + 0.5,
                         "n_tokens": 100 if first == "A" else 50})
            rows.append({"item": item, "pair_uid": f"{item}p", "order": order,
                         "group": "transcript_2", "score": content[second],
                         "n_tokens": 100 if second == "A" else 50})
            rows.append({"item": item, "pair_uid": f"{item}p", "order": order,
                         "group": "question", "score": 0.25, "n_tokens": 10})
    return pd.DataFrame(rows)


def test_span_summary_reports_sum_per_token_and_count():
    table = mask_summary.span_summary(toy_scores(), n_boot=50)

    question = table[(table.group == "question") & (table.quantity == "total")]
    assert np.isclose(question.estimate.iloc[0], 0.25)
    per_token = table[(table.group == "question") & (table.quantity == "per_token")]
    assert np.isclose(per_token.estimate.iloc[0], 0.025)
    assert (table.lo <= table.estimate).all() and (table.estimate <= table.hi).all()


def test_order_asymmetry_recovers_the_position_bonus_and_cancels_content():
    table = mask_summary.order_asymmetry(toy_scores(), n_boot=50)

    def get(quantity, statistic):
        row = table[(table.quantity == quantity) & (table.statistic == statistic)]
        return row.estimate.iloc[0]

    # content (+2 vs -1) swaps slots and cancels in the paired mean; the +0.5
    # position bonus always sits on the first slot and survives
    assert np.isclose(get("total", "first_minus_second"), 0.5)
    # the sum contrast: content also cancels across orders (2 + -1 appears in
    # both orders), leaving the same position bonus
    assert np.isclose(get("total", "net_towards_first"), 0.5 + 1.0)  # bonus + content sum
    assert np.isclose(get("tokens", "first_minus_second"), 0.0)
