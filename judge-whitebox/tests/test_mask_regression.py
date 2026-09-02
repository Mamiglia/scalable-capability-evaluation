import numpy as np
import pandas as pd

import mask_regression
import token_manifold_record


def toy_meta():
    """Two structural groups; the first holds two sentences."""
    prompt = "One two. Three\nfour five"
    #         0123456789...
    words = [(0, 3), (4, 7), (7, 8), (9, 14), (15, 19), (20, 24)]
    return {"prompt": prompt, "input_ids": list(range(6)), "offsets": words,
            "group_id": np.array([0, 0, 0, 0, 1, 1], dtype=np.int8),
            "group_names": token_manifold_record.STRUCTURAL_GROUPS}


def test_sentence_units_split_on_punctuation_and_group_change():
    units = mask_regression.sentence_units(toy_meta())

    # "One two." | "Three\nfour" | group change -> "five ..." — the newline
    # also ends a sentence, so "Three" and "four" separate as well
    assert units.tolist() == [0, 0, 0, 1, 2, 2]


def test_unit_table_reports_group_size_and_text():
    meta = toy_meta()
    table = mask_regression.unit_table(meta, mask_regression.sentence_units(meta))

    assert table.n_tokens.tolist() == [3, 1, 2]
    assert table.text.tolist() == ["One two.", "Three", "four five"]
    assert table.group.tolist() == [token_manifold_record.STRUCTURAL_GROUPS[0]] * 2 + \
        [token_manifold_record.STRUCTURAL_GROUPS[1]]


def test_design_matrix_counts_blanked_tokens_per_unit():
    unit_ids = np.array([0, 0, 1, 1, 2, 2])

    X = mask_regression.design_matrix([[0, 2, 3], [4]], unit_ids)

    assert X.tolist() == [[1, 2, 0], [0, 0, 1]]


def test_fit_units_recovers_planted_effects_with_high_heldout_r2():
    rng = np.random.default_rng(0)
    truth = np.array([2.0, -1.0, 0.5])
    X = rng.integers(0, 4, size=(200, 3)).astype(float)
    y = X @ truth

    fit = mask_regression.fit_units(X, y)

    assert np.allclose(fit["beta"], truth, atol=1e-4)
    assert fit["heldout_r2"] > 0.999


def test_score_campaign_scores_every_prompt_and_carries_the_gate(monkeypatch):
    monkeypatch.setattr(token_manifold_record, "prompt_token_meta",
                        lambda row, tokenizer, c: toy_meta())
    rng = np.random.default_rng(1)
    draws = [rng.choice(6, size=2, replace=False).tolist() for _ in range(64)]
    truth = np.array([1.0, 1.0, 1.0, -2.0, 0.5, 0.5])
    masks = pd.DataFrame({
        "pair_uid": "p0", "item": "it0", "order": 0,
        "draw": range(64), "masked_positions": draws,
        "delta": [truth[d].sum() for d in draws],
        "gap": 0.7, "n_tokens": 6})
    df = pd.DataFrame([{"pair_uid": "p0", "order": 0}])

    out = mask_regression.score_campaign(masks, df, None, {}, "sentence")

    assert len(out) == 3
    assert np.allclose(out.score, [3.0, -2.0, 1.0], atol=1e-3)
    assert (out.heldout_r2 > 0.999).all()
    # group level: token effects inside group 0 are heterogeneous (1,1,1,-2),
    # so counts predict draws poorly and the gate must reflect that
    grouped = mask_regression.score_campaign(masks, df, None, {}, "group")
    assert len(grouped) == 2
    assert np.allclose(grouped.score, [1.0, 1.0], atol=0.3)
    assert grouped.heldout_r2.iloc[0] < out.heldout_r2.iloc[0]


def test_word_units_split_on_whitespace_and_keep_multitoken_words_together():
    # "One two" tokenised as [One][ two-part-a][part-b]: the leading-space token
    # starts a word, the flush-adjacent token continues it
    meta = {"prompt": "One two", "input_ids": [0, 1, 2],
            "offsets": [(0, 3), (3, 6), (6, 7)],
            "group_id": np.array([0, 0, 0], dtype=np.int8),
            "group_names": token_manifold_record.STRUCTURAL_GROUPS}

    assert mask_regression.word_units(meta).tolist() == [0, 1, 1]


def test_token_units_are_the_identity():
    meta = toy_meta()
    assert mask_regression.token_units(meta).tolist() == list(range(6))
