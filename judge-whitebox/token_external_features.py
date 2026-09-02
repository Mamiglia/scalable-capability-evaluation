"""Post-hoc external covariates for anonymous token-manifold objects."""

import numpy as np
import pandas as pd


def external_frame(prompts, pairs, prompt_idx, group_id):
    """Join capability, exact transcript length, position, and judgement errors."""
    meta = pd.DataFrame(prompts).reset_index(names="prompt_idx")
    columns = ["pair_uid", "order", "cap_1", "cap_2", "label", "text_1", "text_2"]
    meta = meta.merge(pairs[columns], on=["pair_uid", "order"], how="left",
                      validate="one_to_one")
    if meta[columns[2:]].isna().any().any():
        raise ValueError("trace prompts did not match pair metadata")
    counts = (pd.DataFrame({"prompt_idx": prompt_idx, "group_id": group_id})
              .groupby(["prompt_idx", "group_id"]).size().unstack(fill_value=0))
    meta["transcript_1_tokens"] = meta.prompt_idx.map(counts.get(3, 0)).astype(int)
    meta["transcript_2_tokens"] = meta.prompt_idx.map(counts.get(4, 0)).astype(int)
    meta["capability_gap"] = (meta.cap_1 - meta.cap_2).round(12)
    meta["character_length_gap"] = meta.text_1.str.len() - meta.text_2.str.len()
    meta["token_length_gap"] = meta.transcript_1_tokens - meta.transcript_2_tokens
    meta["pick"] = np.where(meta.gap > 0, 1, 2)
    meta["correct"] = meta.pick == meta.label
    meta["weaker_slot"] = 3 - meta.label
    meta["longer_slot"] = np.where(meta.token_length_gap > 0, 1,
                                    np.where(meta.token_length_gap < 0, 2, 0))
    meta["picked_longer"] = (meta.longer_slot != 0) & (meta.pick == meta.longer_slot)
    meta["longer_weaker"] = ((meta.longer_slot != 0)
                              & (meta.longer_slot == meta.weaker_slot))
    meta["picked_longer_weaker"] = (meta.picked_longer & meta.longer_weaker
                                      & ~meta.correct)
    meta["picked_slot_1"] = meta.pick == 1
    return meta
