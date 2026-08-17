"""Per-token residual recording for output-conditioned manifold discovery."""

import numpy as np

import token_ablate
import wbprompts


STRUCTURAL_GROUPS = ("instr_format", "slot_labels", "question", "transcript_1",
                     "transcript_2", "answer_readout")


def balanced_item_folds(df, seed):
    """Deterministically split whole items while greedily balancing row counts."""
    counts = df.groupby("item", sort=True).size()
    rng = np.random.default_rng(seed)
    tie = dict(zip(counts.index, rng.random(len(counts))))
    ordered = sorted(counts.index, key=lambda item: (-int(counts[item]), tie[item], str(item)))
    totals = {"A": 0, "B": 0}
    folds = {}
    for item in ordered:
        fold = min(totals, key=lambda name: (totals[name], name))
        folds[item] = fold
        totals[fold] += int(counts[item])
    return folds


def select_fold(df, folds, name):
    """Return one fold without reordering rows or splitting items."""
    assert name in {"A", "B"}
    assert set(df["item"]) <= set(folds)
    return df[df["item"].map(folds) == name].copy()


def layer_indices(n_states, n_select=12):
    """Evenly spaced hidden-state indices including both endpoints."""
    assert n_states > 0 and n_select > 0
    points = np.rint(np.linspace(0, n_states - 1, min(n_states, n_select))).astype(int)
    return tuple(sorted(set(points.tolist())))


def prompt_token_meta(row, tokenizer, c):
    """Aligned token IDs, offsets, and one structural group ID per token."""
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    groups = token_ablate.token_groups(row, tokenizer, c)
    group_id = np.full(len(enc["input_ids"]), -1, dtype=np.int8)
    for idx, name in enumerate(STRUCTURAL_GROUPS):
        group_id[groups[name]] = idx
    assert (group_id >= 0).all(), "structural token groups must cover the prompt"
    return {"prompt": prompt, "input_ids": enc["input_ids"],
            "offsets": enc["offset_mapping"], "group_id": group_id,
            "group_names": STRUCTURAL_GROUPS}


def estimate_storage(df, tokenizer, c, layers, d_model, dtype_bytes=2):
    """Exact residual-byte estimate from rendered token counts; writes nothing."""
    n_tokens = sum(len(tokenizer(wbprompts.render_verdict_prompt(row, tokenizer))["input_ids"])
                   for _, row in df.iterrows())
    return {"n_prompts": len(df), "n_tokens": n_tokens, "n_layers": len(layers),
            "d_model": d_model,
            "residual_bytes": n_tokens * len(layers) * d_model * dtype_bytes,
            "metadata_bytes_upper": n_tokens * 32}
