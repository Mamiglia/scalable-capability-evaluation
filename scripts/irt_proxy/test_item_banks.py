import numpy as np
import pandas as pd

from irt.data import pick_epoch


def test_pick_epoch_keeps_same_item_id_across_benchmarks():
    responses = pd.DataFrame([
        {"model": "m", "benchmark": "aime2025", "item": "1", "epoch": 1, "answer": "7", "correct": True},
        {"model": "m", "benchmark": "aime2026", "item": "1", "epoch": 1, "answer": "8", "correct": False},
    ])

    got = pick_epoch(responses)

    assert set(zip(got["benchmark"], got["item"])) == {
        ("aime2025", "1"),
        ("aime2026", "1"),
    }


def test_fixed_3pl_prediction_respects_supplied_floor():
    from irt.fitting import predict

    posterior = {
        "theta": np.array([[-50.0]]),
        "a": np.array([[1.0]]),
        "b": np.array([[0.0]]),
    }

    gpqa = predict(
        "fixed_3pl", posterior, np.array([0]), np.array([0]),
        lower_asymptote=np.array([0.25]),
    )
    aime = predict(
        "fixed_3pl", posterior, np.array([0]), np.array([0]),
        lower_asymptote=np.array([0.001]),
    )

    assert np.isclose(gpqa[0], 0.25)
    assert np.isclose(aime[0], 0.001)


def test_irt_cells_prefers_expansion_before_epoch_selection():
    from item_banks import irt_cells

    responses = pd.DataFrame([
        {"model": "m", "benchmark": "gpqa", "item": "i", "epoch": 1, "answer": "A", "correct": False, "source_priority": 0},
        {"model": "m", "benchmark": "gpqa", "item": "i", "epoch": 2, "answer": "B", "correct": True, "source_priority": 2},
    ])

    got = irt_cells(responses)

    assert len(got) == 1
    assert bool(got.iloc[0]["correct"])


def test_assign_splits_stratifies_joint_by_benchmark_year():
    from item_banks import assign_splits

    difficulties = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "joint", "benchmark": benchmark, "item": str(i), "difficulty": float(i)}
        for benchmark in ("gpqa", "aime2025", "aime2026")
        for i in range(8)
    ])

    got = assign_splits(difficulties)

    counts = got.groupby(["benchmark", "half"]).size().to_dict()
    assert all(
        counts[(benchmark, half)] == 4
        for benchmark in ("gpqa", "aime2025", "aime2026")
        for half in ("easy", "hard")
    )
    assert set(got.loc[got["quartile"] == "Q1", "item"]) == {"0", "1"}


def test_assign_splits_adds_stable_tertiles_and_quintiles():
    from item_banks import assign_splits

    difficulties = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "benchmark": "gpqa",
            "item": str(i),
            "difficulty": float(i),
        }
        for i in range(10)
    ])

    got = assign_splits(difficulties)

    assert got.groupby("tertile").size().to_dict() == {"T1": 4, "T2": 3, "T3": 3}
    assert got.groupby("quintile").size().to_dict() == {
        "P1": 2,
        "P2": 2,
        "P3": 2,
        "P4": 2,
        "P5": 2,
    }
    assert got.loc[got["tertile"] == "T1", "item"].tolist() == ["0", "1", "2", "3"]


def test_fit_difficulties_runs_declared_models_and_format_floors(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {
            "model": model,
            "benchmark": benchmark,
            "item": "1",
            "epoch": 1,
            "answer": "A",
            "correct": model == "response-only",
            "source_priority": 1,
        }
        for model in ("judged", "response-only")
        for benchmark in ("gpqa", "aime2025", "aime2026")
    ])
    fitted_models = []

    def fake_fit(data, **_kwargs):
        fitted_models.append(set(data.models))
        n_items = len(data.items)
        return {
            "theta": np.zeros((2, len(data.models))),
            "a": np.ones((2, n_items)),
            "b": np.vstack([np.arange(n_items), np.arange(n_items)]),
        }

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    got = item_banks.fit_difficulties(
        responses,
        floors={"gpqa": 0.25, "aime2025": 0.001, "aime2026": 0.001},
        warmup=2,
        samples=2,
        seed=7,
    )

    assert set(got["fit"]) == {"gpqa", "aime", "joint"}
    assert set(got["irt_model"]) == {"fixed_3pl", "2pl"}
    assert set(got.loc[got["benchmark"] == "gpqa", "lower_asymptote"]) == {0.0, 0.25}
    assert set(got.loc[got["benchmark"].str.startswith("aime"), "lower_asymptote"]) == {0.0, 0.001}
    assert all(models == {"judged", "response-only"} for models in fitted_models)


def test_fit_parameters_returns_model_theta_and_gpqa_posterior(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": model, "benchmark": benchmark, "item": item, "epoch": 1,
         "answer": "A", "correct": model == "m1", "source_priority": 1}
        for model in ("m1", "m2")
        for benchmark, item in (
            ("gpqa", "g1"), ("gpqa", "g2"),
            ("aime2025", "a25"), ("aime2026", "a26"),
        )
    ])

    def fake_fit(data, **_kwargs):
        return {
            "theta": np.array([[1.0, -1.0], [1.2, -0.8]]),
            "a": np.ones((2, len(data.items))),
            "b": np.zeros((2, len(data.items))),
        }

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    difficulties, capabilities, posteriors = item_banks.fit_parameters(
        responses, floors={"gpqa": 0.25, "aime2025": 0.001, "aime2026": 0.001},
        warmup=2, samples=2, seed=7,
    )

    gpqa = capabilities.query("irt_model == 'fixed_3pl' and fit == 'gpqa'")
    assert dict(zip(gpqa["model"], gpqa["theta"])) == {"m1": 1.1, "m2": -0.9}
    assert set(difficulties["fit"]) == {"gpqa", "aime", "joint"}
    assert posteriors["fixed_3pl__gpqa"]["models"] == ["m1", "m2"]
