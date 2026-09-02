import re

import pandas as pd

ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:.*$")
ASSISTANT_PREFIX = re.compile(r"^\[Assistant\]:\s*")


def reasoning_chars(text: str) -> int:
    """Characters of transcript content beyond the assistant prefix and answer line."""
    return len(ANSWER_LINE.sub("", ASSISTANT_PREFIX.sub("", text)).strip())


def drop_answer_only_pairs(df: pd.DataFrame, min_chars: int) -> pd.DataFrame:
    """Drop pairs where either transcript carries no reasoning, only its answer.

    Some evaluated models emit just "ANSWER: B"; a judge comparison against an
    empty transcript degenerates, so such pairs are excluded whole.
    """
    keep = (df["text_1"].map(reasoning_chars) >= min_chars) \
        & (df["text_2"].map(reasoning_chars) >= min_chars)
    bad_pairs = set(df.loc[~keep, "pair_uid"])
    return df[~df["pair_uid"].isin(bad_pairs)].copy()


def load_pairs(c: dict, split: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load pairs from parquet, filter by split, limit by whole pair_uid.

    Asserts both orders present per pair_uid and no <think> in texts.
    Returns DataFrame with exact schema: pair_uid, order, item, model_1, model_2,
    cap_1, cap_2, label, question, options_1, options_2, text_1, text_2, split.
    """
    path = c["paths"]["pairs"]
    df = pd.read_parquet(path)

    if split is not None:
        df = df[df["split"] == split].copy()

    # Check for <think> in text columns
    for col in ["text_1", "text_2"]:
        if (df[col].str.contains("<think>", case=False, na=False)).any():
            raise ValueError(f"Found <think> in {col}")

    if limit is not None:
        unique_pairs = df["pair_uid"].unique()[:limit]
        df = df[df["pair_uid"].isin(unique_pairs)].copy()

    # Assert both orders present for each pair_uid
    for pair_uid in df["pair_uid"].unique():
        orders = set(df[df["pair_uid"] == pair_uid]["order"].unique())
        assert orders == {0, 1}, f"pair_uid {pair_uid} missing order(s): {orders}"

    return df


def select_kept_rows(pairs_df: pd.DataFrame, screen_dict: dict) -> pd.DataFrame:
    """Judging-screen rows positionally aligned to pairs_df via kept_row_idx (screen.py drops
    over-length rows, so this isn't just a range). pairs_df must already be loaded with the
    screen's own split/limit. Asserts set_name == "judging" (only judging screens are
    row-per-pair -- control fans one pairs-df row into several batch entries) and that the
    kept-row count matches n_prompts."""
    set_name = screen_dict.get("set_name")
    assert set_name == "judging", (
        f"screen result must be a judging-set screen (rows align 1:1 with pairs_df); "
        f"got set_name={set_name!r}. Re-run screen.py with --set judging.")
    n_prompts = screen_dict["n_prompts"]
    kept_row_idx = screen_dict.get("kept_row_idx")
    if kept_row_idx is None:
        print("WARNING: screen result has no kept_row_idx (pre-task-14 screen.py) -- assuming "
              "no rows were skipped and aligning positionally; only valid if the screen run saw "
              "no over-length prompts (true at smoke scale, not guaranteed at full scale).")
        kept_row_idx = list(range(n_prompts))
    meta = pairs_df.iloc[kept_row_idx].reset_index(drop=True)
    assert len(meta) == n_prompts, (
        f"reconstructed prompt_meta has {len(meta)} rows, expected {n_prompts} -- "
        "split/limit/kept_row_idx don't match this screen result")
    return meta
