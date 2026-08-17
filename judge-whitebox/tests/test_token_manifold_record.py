import numpy as np
import pandas as pd
import torch

import token_manifold_record as record


def fake_row(uid="p0"):
    return {"pair_uid": uid, "order": 0, "item": "it0", "label": 1,
            "question": "Q", "options_1": "", "options_2": "",
            "text_1": "abc", "text_2": "de"}


def fake_cfg():
    return {"token_ablate": {"markers": {}}}


class OffsetTokenizer:
    def __call__(self, text, return_offsets_mapping=False, return_tensors=None):
        ids = [1 + i % 30 for i in range(len(text))]
        out = {"input_ids": torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


LengthTokenizer = OffsetTokenizer


def test_balanced_item_folds_are_deterministic_disjoint_and_row_balanced():
    df = pd.DataFrame({"item": ["heavy"] * 8 + ["a"] * 4 + ["b"] * 4 + ["c"] * 2})

    one = record.balanced_item_folds(df, seed=7)
    two = record.balanced_item_folds(df.sample(frac=1, random_state=9), seed=7)

    assert one == two
    assert set(one) == set(df.item)
    assert set(one.values()) == {"A", "B"}
    counts = df.assign(fold=df.item.map(one)).groupby("fold").size()
    assert abs(int(counts.A) - int(counts.B)) <= 2


def test_select_fold_keeps_whole_items_and_original_order():
    df = pd.DataFrame({"item": ["a", "a", "b", "b"], "row": range(4)})

    out = record.select_fold(df, {"a": "B", "b": "A"}, "A")

    assert out.row.tolist() == [2, 3]
    assert set(out.item) == {"b"}


def test_layer_indices_include_ends_and_are_unique():
    assert record.layer_indices(33, 4) == (0, 11, 21, 32)
    assert record.layer_indices(3, 12) == (0, 1, 2)


def test_prompt_token_meta_aligns_ids_offsets_and_structural_groups(monkeypatch):
    row, tok, cfg = fake_row(), OffsetTokenizer(), fake_cfg()
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: "QABCD")
    monkeypatch.setattr(record.token_ablate, "token_groups", lambda row, tok, c: {
        "question": [0], "transcript_1": [1], "transcript_2": [2],
        "instr_format": [3], "slot_labels": [], "answer_readout": [4],
    })

    meta = record.prompt_token_meta(row, tok, cfg)

    assert len(meta["input_ids"]) == len(meta["offsets"]) == len(meta["group_id"]) == 5
    assert meta["group_names"][meta["group_id"][1]] == "transcript_1"


def test_estimate_storage_counts_rendered_tokens_and_residual_bytes(monkeypatch):
    df = pd.DataFrame([fake_row("p0"), fake_row("p1")])
    monkeypatch.setattr(record.wbprompts, "render_verdict_prompt", lambda row, tok: row["text_1"])

    out = record.estimate_storage(df, LengthTokenizer(), fake_cfg(), layers=(0, 2), d_model=4)

    assert out["n_prompts"] == 2
    assert out["n_tokens"] == len(df.iloc[0].text_1) + len(df.iloc[1].text_1)
    assert out["residual_bytes"] == out["n_tokens"] * 2 * 4 * 2
