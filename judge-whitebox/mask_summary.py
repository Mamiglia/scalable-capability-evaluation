"""Descriptive summaries of unit scores: where the effect sits.

Consumes the mask_regression output table. Every interval is a bootstrap over
items — the unit of independence — never over prompts or units. Scores are
reported as signed sums, mean per token and token count, never shares: a share
would renormalise the length effect away, and length is a result here.

Sign convention: positive score means blanking the unit lowered F = z1 − z2,
i.e. the unit was pushing towards answer 1.
"""

import numpy as np
import pandas as pd


def item_bootstrap(frame, stat, n_boot=2000, seed=0):
    """Percentile interval for stat(frame) under resampling of whole items."""
    items = frame["item"].unique()
    by_item = dict(tuple(frame.groupby("item")))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(items, size=len(items), replace=True)
        draws.append(stat(pd.concat([by_item[i] for i in sample], ignore_index=True)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(frame)), "lo": float(lo), "hi": float(hi)}


def span_summary(scores, n_boot=2000, seed=0):
    """Per structural group: signed sum, mean per token, token count, with intervals.

    Sums are per prompt, then averaged across prompts, so long prompts do not
    dominate through their unit count.
    """
    rows = []
    for group, g in scores.groupby("group"):
        per_prompt = g.groupby(["item", "pair_uid", "order"]).agg(
            total=("score", "sum"), tokens=("n_tokens", "sum")).reset_index()
        per_prompt["per_token"] = per_prompt["total"] / per_prompt["tokens"]
        for name in ("total", "per_token", "tokens"):
            ci = item_bootstrap(per_prompt, lambda f, c=name: f[c].mean(), n_boot, seed)
            rows.append({"group": group, "quantity": name, **ci})
    return pd.DataFrame(rows)


def order_asymmetry(scores, n_boot=2000, seed=0):
    """First-presented minus second-presented transcript, paired within pair.

    The same two transcripts swap slots between the pair's two orders, so this
    difference isolates position from content. Run on the summed score, the
    mean per token and the token count; a position effect shows in the first
    two, a length imbalance only in the third.
    """
    slots = scores[scores["group"].isin(["transcript_1", "transcript_2"])]
    per_slot = slots.groupby(["item", "pair_uid", "order", "group"]).agg(
        total=("score", "sum"), tokens=("n_tokens", "sum")).reset_index()
    per_slot["per_token"] = per_slot["total"] / per_slot["tokens"]
    wide = per_slot.pivot_table(index=["item", "pair_uid", "order"], columns="group",
                                values=["total", "per_token", "tokens"])
    rows = []
    # Scores carry the answer's sign (positive = towards answer 1), so two
    # contrasts are needed: the difference measures how much transcripts
    # advocate for their own slot; the sum is the net tilt towards the
    # first-presented slot, with content cancelling across the two orders.
    for name in ("total", "per_token", "tokens"):
        both = wide[(name, "transcript_1")], wide[(name, "transcript_2")]
        for stat, values in (("first_minus_second", both[0] - both[1]),
                             ("net_towards_first", both[0] + both[1])):
            frame = values.reset_index()
            frame.columns = ["item", "pair_uid", "order", "value"]
            paired = frame.groupby(["item", "pair_uid"])["value"].mean().reset_index()
            ci = item_bootstrap(paired, lambda f: f["value"].mean(), n_boot, seed)
            rows.append({"quantity": name, "statistic": stat, **ci})
    return pd.DataFrame(rows)
