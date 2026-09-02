import numpy as np
import pandas as pd
import pytest

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


def test_reference_bank_excludes_every_candidate(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": model, "benchmark": "gpqa", "item": item, "epoch": 1,
         "answer": "A", "correct": model == "reference", "source_priority": 1}
        for model in ("reference", "candidate") for item in ("i1", "i2")
    ])
    seen = {}

    def fake_fit(data, **_kwargs):
        seen["models"] = list(data.models)
        return {"theta": np.zeros((2, 1)), "a": np.ones((2, 2)),
                "b": np.zeros((2, 2))}

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    item_banks.fit_reference_bank(
        responses, {"candidate"}, floor=0.25, irt_model="fixed_3pl",
        warmup=2, samples=2, seed=0,
    )

    assert seen["models"] == ["reference"]


def test_reference_bank_forwards_four_production_chains(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": "reference", "benchmark": "gpqa", "item": item, "epoch": 1,
         "answer": "A", "correct": True, "source_priority": 1}
        for item in ("i1", "i2")
    ])
    seen = {}

    def fake_fit(data, **kwargs):
        seen.update(kwargs)
        return {
            "theta": np.zeros((2, 1)), "a": np.ones((2, 2)),
            "b": np.zeros((2, 2)),
        }

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    item_banks.fit_reference_bank(
        responses, set(), floor=0.25, irt_model="fixed_3pl",
        warmup=2, samples=2, seed=0, chains=4,
    )

    assert seen["chains"] == 4


def test_irt_fit_uses_requested_parallel_chains(monkeypatch):
    from irt.data import IRTData
    import irt.fitting as fitting

    captured = {}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kwargs)

        def run(self, key, **kwargs):
            captured["run"] = kwargs

        def get_samples(self):
            return {"theta": np.zeros((2, 1))}

    monkeypatch.setattr(fitting, "NUTS", lambda model: model)
    monkeypatch.setattr(fitting, "MCMC", FakeMCMC)
    data = IRTData(
        model_idx=np.array([0]), item_idx=np.array([0]), y=np.array([1]),
        models=["m"], items=[("gpqa", "i")],
    )

    fitting.fit(data, model="2pl", warmup=2, samples=2, chains=4)

    assert captured["num_chains"] == 4
    assert captured["chain_method"] == "parallel"
    assert "chains" not in captured["run"]


def test_irt_fit_can_return_chain_diagnostics(monkeypatch):
    from irt.data import IRTData
    import irt.fitting as fitting

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

        def run(self, key, **kwargs):
            pass

        def get_samples(self, group_by_chain=False):
            shape = (4, 2, 1) if group_by_chain else (8, 1)
            return {name: np.zeros(shape) for name in ("theta", "a", "b")}

        def get_extra_fields(self, group_by_chain=False):
            assert group_by_chain
            return {"diverging": np.array([
                [False, False], [False, True], [False, False], [False, False],
            ])}

    monkeypatch.setattr(fitting, "NUTS", lambda model: model)
    monkeypatch.setattr(fitting, "MCMC", FakeMCMC)
    monkeypatch.setattr(
        fitting.numpyro.diagnostics, "summary",
        lambda samples, group_by_chain: {
            "b": {"r_hat": np.array([1.0]), "n_eff": np.array([8.0])},
        },
    )
    data = IRTData(
        model_idx=np.array([0]), item_idx=np.array([0]), y=np.array([1]),
        models=["m"], items=[("gpqa", "i")],
    )

    got = fitting.fit(
        data, model="2pl", warmup=2, samples=2, chains=4,
        collect_diagnostics=True,
    )

    assert got["divergences"] == 1
    assert got["diagnostics"]["b"]["r_hat"] == 1.0


def test_reference_bank_preserves_irt_diagnostics(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": "reference", "benchmark": "gpqa", "item": item, "epoch": 1,
         "answer": "A", "correct": True, "source_priority": 1}
        for item in ("i1", "i2")
    ])

    def fake_fit(data, **kwargs):
        return {
            "theta": np.zeros((2, 1)), "a": np.ones((2, 2)),
            "b": np.zeros((2, 2)), "divergences": 0,
            "diagnostics": {
                "b": {"r_hat": np.array([1.0]), "n_eff": np.array([8.0])},
            },
        }

    monkeypatch.setattr(item_banks, "fit", fake_fit)
    _, artifact = item_banks.fit_reference_bank(
        responses, set(), floor=0.25, irt_model="fixed_3pl",
        warmup=2, samples=2, seed=0, chains=4, collect_diagnostics=True,
    )

    assert artifact["divergences"] == 0
    assert artifact["diagnostics"]["b"]["n_eff"] == 8.0


def test_fixed_items_score_more_correct_model_higher():
    from item_banks import score_fixed_item_capabilities

    responses = pd.DataFrame([
        {"model": model, "benchmark": "gpqa", "item": item,
         "answer": "A", "correct": correct}
        for model, values in (("strong", (1, 1, 1)), ("weak", (1, 0, 0)))
        for item, correct in zip(("i1", "i2", "i3"), values)
    ])
    posterior = {
        "a": np.ones((4, 3)), "b": np.zeros((4, 3)),
        "floor": np.full(3, 0.25), "items": [("gpqa", f"i{i}") for i in (1, 2, 3)],
    }

    got = score_fixed_item_capabilities(responses, posterior, max_draws=4)

    assert got.set_index("model").loc["strong", "theta"] > got.set_index("model").loc["weak", "theta"]


def test_fixed_items_require_two_reference_bank_items():
    from item_banks import score_fixed_item_capabilities

    responses = pd.DataFrame([
        {"model": "candidate", "benchmark": "gpqa", "item": "i1", "answer": "A", "correct": True},
    ])
    posterior = {
        "a": np.ones((1, 2)), "b": np.zeros((1, 2)),
        "floor": np.full(2, 0.25), "items": [("gpqa", "i1"), ("gpqa", "i2")],
    }

    with pytest.raises(ValueError, match="fewer than two"):
        score_fixed_item_capabilities(responses, posterior)


def test_fixed_items_score_canonical_source_priority_and_epoch(monkeypatch):
    import item_banks

    responses = pd.DataFrame([
        {"model": "candidate", "benchmark": "gpqa", "item": "i1",
         "epoch": 1, "answer": "A", "correct": False, "source_priority": 1},
        {"model": "candidate", "benchmark": "gpqa", "item": "i1",
         "epoch": 0, "answer": "A", "correct": False, "source_priority": 2},
        {"model": "candidate", "benchmark": "gpqa", "item": "i1",
         "epoch": 1, "answer": "A", "correct": True, "source_priority": 2},
        {"model": "candidate", "benchmark": "gpqa", "item": "i1",
         "epoch": 2, "answer": "A", "correct": False, "source_priority": 2},
        {"model": "candidate", "benchmark": "gpqa", "item": "i2",
         "epoch": 1, "answer": "B", "correct": False, "source_priority": 2},
    ])
    posterior = {
        "a": np.ones((2, 2)), "b": np.zeros((2, 2)),
        "floor": np.full(2, 0.25),
        "items": [("gpqa", "i1"), ("gpqa", "i2")],
    }
    seen = []

    def fake_theta(correct, *_args):
        seen.append(correct.tolist())
        return 0.0

    monkeypatch.setattr(item_banks, "_theta_map", fake_theta)

    item_banks.score_fixed_item_capabilities(responses, posterior, max_draws=2)

    assert seen == [[True, False], [True, False]]


def test_fixed_items_reject_nonunique_canonical_cells(monkeypatch):
    import item_banks

    duplicated = pd.DataFrame([
        {"model": "candidate", "benchmark": "gpqa", "item": item,
         "answer": "A", "correct": True}
        for item in ("i1", "i1", "i2")
    ])
    monkeypatch.setattr(item_banks, "irt_cells", lambda _: duplicated)
    posterior = {
        "a": np.ones((1, 2)), "b": np.zeros((1, 2)),
        "floor": np.full(2, 0.25),
        "items": [("gpqa", "i1"), ("gpqa", "i2")],
    }

    with pytest.raises(ValueError, match="one canonical response"):
        item_banks.score_fixed_item_capabilities(duplicated, posterior)
