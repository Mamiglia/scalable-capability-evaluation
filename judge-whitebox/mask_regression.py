"""Recover unit scores from random-mask draws, gated by held-out prediction.

A campaign records, per prompt, many draws of (which token positions were
blanked, how far the answer-logit gap moved). Regressing the gap change on
per-unit blanked-token counts gives each unit's score — its per-token push on
the gap times its size — as the best additive fit. The fit is only reported
with its held-out R² over draws: a score table whose fit cannot predict
unseen blankings supports nothing.

Units are structural groups (instructions, question, the two transcripts…)
or sentences within them. Sentences are split on terminal punctuation inside
each structural span, so a sentence never crosses a span boundary.
"""

import pathlib
import re

import numpy as np
import pandas as pd

import token_manifold_record

SENTENCE_END = re.compile(r"[.!?;](?:\s+|$)|\n+")


def sentence_units(meta):
    """Unit id per token: sentences within structural spans.

    Boundaries come from terminal punctuation in the rendered prompt text;
    a new structural group always starts a new unit, so spans keep their
    identity even without punctuation.
    """
    prompt, offsets, group_id = meta["prompt"], meta["offsets"], meta["group_id"]
    ends = {m.end() for m in SENTENCE_END.finditer(prompt)}
    unit_ids = np.zeros(len(offsets), dtype=int)
    unit = 0
    for t in range(1, len(offsets)):
        new_group = group_id[t] != group_id[t - 1]
        crossed_end = any(offsets[t - 1][1] <= e <= offsets[t][0] for e in ends)
        if new_group or crossed_end:
            unit += 1
        unit_ids[t] = unit
    return unit_ids


def group_units(meta):
    """Unit id per token: the structural group itself."""
    return np.asarray(meta["group_id"], dtype=int)


def word_units(meta):
    """Unit id per token: whitespace-delimited words within structural spans.

    A token starts a new word when its group changes, whitespace separates it
    from the previous token, or it begins with whitespace itself (BPE tokens
    carry their leading space).
    """
    prompt, offsets, group_id = meta["prompt"], meta["offsets"], meta["group_id"]
    unit_ids = np.zeros(len(offsets), dtype=int)
    unit = 0
    for t in range(1, len(offsets)):
        gap = prompt[offsets[t - 1][1]:offsets[t][0]]
        tok = prompt[offsets[t][0]:offsets[t][1]]
        if group_id[t] != group_id[t - 1] or any(c.isspace() for c in gap) or tok[:1].isspace():
            unit += 1
        unit_ids[t] = unit
    return unit_ids


def token_units(meta):
    """Unit id per token: every token its own unit."""
    return np.arange(len(meta["offsets"]))


UNIT_FUNCTIONS = {"sentence": sentence_units, "group": group_units,
                  "word": word_units, "token": token_units}


def unit_table(meta, unit_ids):
    """One row per unit: its group, text and token count."""
    groups = meta["group_names"]
    rows = []
    for u in range(int(unit_ids.max()) + 1):
        tokens = np.flatnonzero(unit_ids == u)
        lo, hi = meta["offsets"][tokens[0]][0], meta["offsets"][tokens[-1]][1]
        rows.append({"unit": u, "group": groups[int(meta["group_id"][tokens[0]])],
                     "n_tokens": len(tokens),
                     "text": meta["prompt"][lo:hi]})
    return pd.DataFrame(rows)


def design_matrix(masked_positions, unit_ids):
    """Draws × units matrix of blanked-token counts."""
    n_units = int(unit_ids.max()) + 1
    X = np.zeros((len(masked_positions), n_units))
    for r, positions in enumerate(masked_positions):
        np.add.at(X[r], unit_ids[np.asarray(positions, dtype=int)], 1)
    return X


RIDGE_GRID = tuple(10.0 ** e for e in range(-4, 5))


def _ridge_solve(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def fit_units(X, y, folds=4, seed=0, ridge_grid=RIDGE_GRID, inner_folds=3):
    """Per-unit coefficients and held-out R² over draws, ridge chosen by inner CV.

    Coefficients are per blanked token; a unit's score is coefficient times
    token count. The penalty is selected on inner folds of each training set,
    never on the held-out draws. A penalty chosen at a grid endpoint is
    flagged: it means the grid chose it, not cross-validation.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    def choose(train):
        errs = []
        for lam in ridge_grid:
            err = 0.0
            for g in range(inner_folds):
                itest = train[g::inner_folds]
                itrain = np.setdiff1d(train, itest)
                beta = _ridge_solve(X[itrain], y[itrain], lam)
                err += np.sum((y[itest] - X[itest] @ beta) ** 2)
            errs.append(err)
        return ridge_grid[int(np.argmin(errs))]

    pred = np.empty(n)
    chosen = []
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        lam = choose(train)
        chosen.append(lam)
        pred[test] = X[test] @ _ridge_solve(X[train], y[train], lam)
    heldout_r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    final_lam = choose(order)
    return {"beta": _ridge_solve(X, y, final_lam), "heldout_r2": float(heldout_r2),
            "ridge": float(final_lam),
            "ridge_endpoint": final_lam in (ridge_grid[0], ridge_grid[-1])
            or any(l in (ridge_grid[0], ridge_grid[-1]) for l in chosen)}


def score_prompt(mask_rows, meta, level, folds=4, seed=0):
    """Score one prompt's units from its draws, with the held-out gate attached."""
    unit_ids = UNIT_FUNCTIONS[level](meta)
    X = design_matrix(mask_rows["masked_positions"].tolist(), unit_ids)
    y = mask_rows["delta"].to_numpy(dtype=float)
    fit = fit_units(X, y, folds, seed)
    units = unit_table(meta, unit_ids)
    units["beta_per_token"] = fit["beta"][units["unit"]]
    units["score"] = units["beta_per_token"] * units["n_tokens"]
    units["heldout_r2"] = fit["heldout_r2"]
    units["ridge"] = fit["ridge"]
    units["ridge_endpoint"] = fit["ridge_endpoint"]
    return units


def score_campaign(masks, df, tokenizer, c, level, folds=4, seed=0):
    """Long table of unit scores for every prompt present in both inputs."""
    keyed = {(str(row["pair_uid"]), int(row["order"])): row for _, row in df.iterrows()}
    frames = []
    for (uid, order), rows in masks.groupby(["pair_uid", "order"]):
        row = keyed.get((str(uid), int(order)))
        if row is None:
            continue
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        assert int(rows["n_tokens"].iloc[0]) == len(meta["input_ids"]), \
            "mask table and rendered prompt disagree on length"
        units = score_prompt(rows, meta, level, folds, seed)
        units.insert(0, "pair_uid", uid)
        units.insert(1, "item", rows["item"].iloc[0])
        units.insert(2, "order", int(order))
        units["gap"] = rows["gap"].iloc[0]
        frames.append(units)
    assert frames, "no overlapping prompts between masks and pairs"
    return pd.concat(frames, ignore_index=True)


def read_mask_parts(path):
    """Every part of a campaign shard directory tree in one table."""
    parts = sorted(pathlib.Path(path).rglob("part_*.parquet"))
    assert parts, f"no parts under {path}"
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def main():
    import argparse

    import wbconfig
    import wbpairs
    from transformers import AutoTokenizer

    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", required=True, help="campaign output directory")
    parser.add_argument("--tokenizer", required=True,
                        help="tokenizer name or path matching the campaign model")
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--level", default="sentence",
                        choices=("sentence", "group", "word", "token"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    masks = read_mask_parts(args.masks)
    df = wbpairs.load_pairs(c, args.split)
    # keep only the campaign's own pairs: score_campaign intersects anyway, and
    # holding the full split's transcripts per process exhausts memory
    df = df[df["pair_uid"].isin(masks["pair_uid"].unique())]
    table = score_campaign(masks, df, tokenizer, c, args.level, args.folds)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"{table.pair_uid.nunique()} pairs, "
          f"median held-out R² "
          f"{table.groupby(['pair_uid', 'order']).heldout_r2.first().median():.3f} "
          f"-> {out}")


if __name__ == "__main__":
    main()
