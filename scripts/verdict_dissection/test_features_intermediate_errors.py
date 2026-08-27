import pandas as pd

from features_intermediate_errors import error_features


def test_error_features_count_kinds_and_skip_unparsed():
    records = {"a": {"parsed": True, "errors": [{"quote": "x", "kind": "notation"}, {"quote": "y", "kind": "weird"}]},
               "b": {"parsed": True, "errors": []},
               "c": {"parsed": False, "errors": []}}
    f = error_features(records, pd.Series({"a": 200, "b": 50}))
    assert list(f.index) == ["a", "b"]
    assert f.loc["a", "error_count"] == 2 and f.loc["a", "errors_notation"] == 1 and f.loc["a", "errors_other"] == 1
    assert f.loc["a", "errors_per_100_words"] == 1.0 and f.loc["b", "any_error"] == 0
