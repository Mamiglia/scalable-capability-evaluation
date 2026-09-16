"""Tests for protocols.embeddings: no network, no torch, precomputed .npz stores only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from capability import task
from protocols import embeddings


def toy_task():
    """Like tests/test_protocols.py::toy_task, but each model gets its own log path."""
    items = pd.DataFrame({"bench": "b", "item": ["0", "1"], "guess": 0.0, "a": 1.0, "b": 0.0})
    train_models = list("abc")
    train = task.Cohort(pd.DataFrame({"theta": [0.0, 1.0, 2.0],
                                      "path_b": [f"b/h/{m}/0/log.eval" for m in train_models]},
                                     index=pd.Index(train_models, name="model")), items, "s")
    test_models = list("tu")
    test = task.Cohort(pd.DataFrame({"theta": np.nan,
                                     "path_b": [f"b/h/{m}/0/log.eval" for m in test_models]},
                                    index=pd.Index(test_models, name="model")), items, "s")
    historical = task.Cohort(train.models.iloc[:0], items, "s")
    return task.Task("s", train, historical, test, pd.Series([3.0, 4.0], index=test.models.index))


def write_store(rel_log, ids, epochs, correct, first_dim):
    """One embedding .npz: column 0 = first_dim (scalar or per-row), columns 1-3 constant."""
    n = len(ids)
    embedding = np.empty((n, 4), dtype=np.float16)
    embedding[:, 0] = first_dim
    embedding[:, 1], embedding[:, 2], embedding[:, 3] = 0.5, 0.25, 0.125
    path = embeddings.embedding_path(rel_log, "full")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=str), epochs=np.array(epochs, np.int32), correct=np.array(correct, bool),
             n_tokens=np.ones(n, np.int32), n_chunks=np.ones(n, np.int32), embedding=embedding,
             meta=np.array('{"variant": "full"}'))


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Embedding files for a, b, c, t (column 0 = theta + 1) and correct-transcript listings for every train item."""
    monkeypatch.setattr(embeddings, "EMBEDDINGS", tmp_path)
    embeddings.read_embeddings.cache_clear()
    for model, first_dim in zip("abct", [1.0, 2.0, 3.0, 4.0]):
        write_store(f"b/h/{model}/0/log.eval", ["0", "1", "2"], [1, 1, 2], [True, True, True], first_dim)
    # Cohort.transcripts reads the logs; here every model solved every train item on its first attempt.
    monkeypatch.setattr(task, "read_transcripts", lambda path, require_correct, items: {i: [object()] for i in items})
    return tmp_path


def test_features_are_mean_vectors_and_the_registered_estimators_forecast(stores):
    t = toy_task()
    table = embeddings.EmbeddingRegression("full").get_features(t.train)
    assert list(table.index) == list("abc") and np.allclose(table["0"], [1.0, 2.0, 3.0])
    # theta* = column0 - 1 exactly; t follows the same pattern, u has no embedding file at all
    theta_hat = embeddings.EmbeddingRegression("full", "lin", LinearRegression())(t)
    assert np.isclose(theta_hat["t"], 3.0) and np.isnan(theta_hat["u"])
    for name in ("emb_kr", "emb_knn", "emb_ens"):
        forecast = embeddings.PROTOCOLS[name](t)
        assert np.isfinite(forecast["t"]) and 0.0 <= forecast["t"] <= 2.0 and np.isnan(forecast["u"]), name


def test_read_embeddings_keeps_correct_first_attempts_only(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "EMBEDDINGS", tmp_path)
    embeddings.read_embeddings.cache_clear()
    rel_log = "b/h/m/0/log.eval"
    write_store(rel_log, ["0", "1", "2", "3"], [1, 1, 2, 1], [True, False, True, True], [10.0, 20.0, 30.0, 40.0])
    table = embeddings.read_embeddings(embeddings.embedding_path(rel_log, "full"))
    assert list(table.index) == ["0", "3"] and table.loc["0", 0] == 10.0
    assert embeddings.read_embeddings(embeddings.embedding_path("b/h/none/0/log.eval", "full")).empty


def test_missing_root_forecasts_nothing_and_bad_variant_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "EMBEDDINGS", tmp_path / "none")
    embeddings.read_embeddings.cache_clear()
    monkeypatch.setattr(task, "read_transcripts", lambda path, require_correct, items: {i: [object()] for i in items})
    assert embeddings.EmbeddingRegression("full")(toy_task()).isna().all()
    with pytest.raises(AssertionError):
        embeddings.EmbeddingRegression("bogus")
