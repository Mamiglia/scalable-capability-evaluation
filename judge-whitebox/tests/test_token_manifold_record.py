import pandas as pd

import token_manifold_record as record


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
