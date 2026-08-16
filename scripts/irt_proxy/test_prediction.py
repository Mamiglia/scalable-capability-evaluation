import numpy as np
import pandas as pd


def test_item_votes_counterbalances_within_collection():
    from scores import item_votes

    votes = pd.DataFrame([
        {"source": "s", "benchmark": "gpqa", "item_id": "i", "judge": "j", "major": "a", "minor": "b", "order": "1", "picked_major": 1.0, "n_calls": 1},
        {"source": "s", "benchmark": "gpqa", "item_id": "i", "judge": "j", "major": "a", "minor": "b", "order": "2", "picked_major": 0.0, "n_calls": 1},
    ])

    got = item_votes(votes).iloc[0]

    assert got["picked_major"] == 0.5
    assert got["n_calls"] == 2
    assert got["vote_id"] == "s:i"


def test_filter_judges_uses_pair_macro_accuracy_above_chance():
    from scores import filter_judges_above_chance

    rounds = pd.DataFrame([
        # Ten wins on one pair and one loss on another is excellent when item
        # weighted, but exactly chance when each candidate pair counts once.
        *[
            {"benchmark": "gpqa", "judge": "chance", "major": "a", "minor": "b",
             "picked_major": 1.0}
            for _ in range(10)
        ],
        {"benchmark": "gpqa", "judge": "chance", "major": "c", "minor": "d",
         "picked_major": 1.0},
        {"benchmark": "gpqa", "judge": "useful", "major": "a", "minor": "b",
         "picked_major": 1.0},
        {"benchmark": "gpqa", "judge": "useful", "major": "c", "minor": "d",
         "picked_major": 0.5},
        # No ECI exists for these candidates, so this pair cannot define judge
        # quality and must not rescue the chance-level judge.
        {"benchmark": "gpqa", "judge": "chance", "major": "x", "minor": "y",
         "picked_major": 1.0},
    ])
    metadata = pd.DataFrame([
        {"model": "a", "eci": 150.0},
        {"model": "b", "eci": 100.0},
        {"model": "c", "eci": 90.0},
        {"model": "d", "eci": 120.0},
    ])

    kept, diagnostics = filter_judges_above_chance(rounds, metadata)

    assert set(kept["judge"]) == {"useful"}
    result = diagnostics.set_index("judge")
    assert result.loc["chance", "mean_pair_agreement_with_higher_eci"] == 0.5
    assert result.loc["chance", "n_eci_model_pairs"] == 2
    assert not bool(result.loc["chance", "included"])
    assert result.loc["useful", "mean_pair_agreement_with_higher_eci"] == 0.75
    assert bool(result.loc["useful", "included"])


def test_model_accuracy_weights_items_and_strata_equally():
    from scores import model_accuracy

    responses = pd.DataFrame([
        {"model": "m", "benchmark": "gpqa", "item": "g1", "correct": True, "answer": "A"},
        {"model": "m", "benchmark": "gpqa", "item": "g2", "correct": False, "answer": "B"},
        {"model": "m", "benchmark": "aime2025", "item": "a1", "correct": False, "answer": "3"},
    ])
    items = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "joint", "benchmark": "gpqa", "item": "g1", "subset": "hard"},
        {"irt_model": "fixed_3pl", "fit": "joint", "benchmark": "gpqa", "item": "g2", "subset": "hard"},
        {"irt_model": "fixed_3pl", "fit": "joint", "benchmark": "aime2025", "item": "a1", "subset": "hard"},
    ])

    got = model_accuracy(responses, items).iloc[0]

    assert got["accuracy"] == 0.25
    assert got["n_items"] == 3
    assert got["n_attempts"] == 3


def test_model_accuracy_rejects_repeated_epochs():
    import pytest
    from scores import model_accuracy

    responses = pd.DataFrame([
        {"model": "m", "benchmark": "gpqa", "item": "i", "correct": value,
         "answer": "A", "epoch": epoch}
        for epoch, value in ((1, True), (2, False))
    ])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "subset": "easy",
    }])

    with pytest.raises(ValueError, match="one canonical response"):
        model_accuracy(responses, items)


def test_easy_elos_report_every_connected_component():
    from metrics import rank_metrics
    from scores import analysis_frame, fit_easy_elos

    rounds = pd.DataFrame([
        {"source": "s", "benchmark": "gpqa", "item_id": "i1", "vote_id": "s:i1", "judge": "j", "major": "a", "minor": "b", "picked_major": 1.0, "n_calls": 2},
        {"source": "s", "benchmark": "gpqa", "item_id": "i1", "vote_id": "s:i1", "judge": "j", "major": "b", "minor": "c", "picked_major": 1.0, "n_calls": 2},
        {"source": "s", "benchmark": "gpqa", "item_id": "i2", "vote_id": "s:i2", "judge": "j", "major": "x", "minor": "y", "picked_major": 1.0, "n_calls": 2},
    ])
    splits = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa", "item": item, "half": "easy", "quartile": "Q1"}
        for item in ("i1", "i2")
    ])
    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": subset, "model": model, "accuracy": accuracy}
        for subset, accuracy in (("easy", 0.8), ("hard", 0.4))
        for model in ("a", "b", "c", "x", "y")
    ])

    elos, coverage = fit_easy_elos(rounds, splits)
    frame = analysis_frame(elos, accuracies, coverage)
    judged = frame[frame["judge"] == "j"]
    judged_coverage = coverage[coverage["judge"] == "j"]

    assert set(judged["model"]) == {"a", "b", "c", "x", "y"}
    assert judged.groupby("component")["model"].nunique().sort_values().tolist() == [2, 3]
    assert judged_coverage["n_component_models"].sort_values().tolist() == [2, 3]
    eligible = rank_metrics(judged).query("n_models == 3")
    assert set(eligible["n_component_models"]) == {3}
    assert set(eligible["n_rounds"]) == {2}


def test_component_ids_are_anchored_to_the_pooled_easy_graph():
    from scores import fit_easy_elos

    # The component sizes swap between judges. Size-only numbering would call
    # the a/b/c component C0 for j1 but C1 for j2.
    rounds = pd.DataFrame([
        {"source": "s", "benchmark": "gpqa", "item_id": "i", "vote_id": "s:i", "judge": judge, "major": major, "minor": minor, "picked_major": 1.0, "n_calls": 1}
        for judge, pairs in (
            ("j1", (("a", "b"), ("b", "c"), ("x", "y"))),
            ("j2", (("a", "b"), ("x", "y"), ("y", "z"))),
        )
        for major, minor in pairs
    ])
    splits = pd.DataFrame([{
        "irt_model": "fixed_3pl",
        "fit": "gpqa",
        "benchmark": "gpqa",
        "item": "i",
        "half": "easy",
        "quartile": "Q1",
    }])

    elos, _ = fit_easy_elos(rounds, splits)

    per_judge = elos[elos["judge"].isin(["j1", "j2"])]
    component_for = lambda judge, model: per_judge.loc[
        (per_judge["judge"] == judge) & (per_judge["model"] == model),
        "component",
    ].iloc[0]
    assert component_for("j1", "a") == component_for("j2", "a")
    assert component_for("j1", "x") == component_for("j2", "x")
    assert component_for("j1", "a") != component_for("j1", "x")


def test_easy_elos_can_filter_tertile_and_quintile_banks():
    from scores import fit_easy_elos

    rounds = pd.DataFrame([
        {
            "source": "s",
            "benchmark": "gpqa",
            "item_id": item,
            "vote_id": f"s:{item}",
            "judge": "j",
            "major": major,
            "minor": minor,
            "picked_major": 1.0,
            "n_calls": 1,
        }
        for item, major, minor in (("easy", "a", "b"), ("later", "x", "y"))
    ])
    splits = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "benchmark": "gpqa",
            "item": item,
            "half": "easy",
            "quartile": quartile,
            "tertile": tertile,
            "quintile": quintile,
        }
        for item, quartile, tertile, quintile in (
            ("easy", "Q1", "T1", "P1"),
            ("later", "Q2", "T2", "P2"),
        )
    ])

    for bank in ("T1", "P1"):
        elos, _ = fit_easy_elos(rounds, splits, bank=bank)
        assert set(elos["model"]) == {"a", "b"}


def test_predictor_filter_uses_strict_ninety_percent_cutoff():
    from scores import select_predictor_models

    values = [0.95] * 19 + [0.85] * 3 + [0.80] + [0.78] * 2
    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": "Q1",
         "model": f"m{index:02d}", "accuracy": accuracy}
        for index, accuracy in enumerate(values)
    ])
    candidates = accuracies[["irt_model", "fit", "model"]]

    eligible, selection = select_predictor_models(
        accuracies,
        candidates,
        "Q1",
    )

    assert len(eligible) == 19
    assert selection.iloc[0]["predictor_accuracy_threshold"] == 0.90
    assert selection.iloc[0]["n_threshold_candidates"] == 19
    assert bool(selection.iloc[0]["threshold_target_met"])


def test_predictor_filter_does_not_relax_when_few_models_clear_ninety_percent():
    from scores import select_predictor_models

    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": "P1",
         "model": model, "accuracy": accuracy}
        for model, accuracy in (("a", 0.95), ("b", 0.85), ("c", 0.80), ("d", 0.76))
    ])

    eligible, selection = select_predictor_models(
        accuracies,
        accuracies[["irt_model", "fit", "model"]],
        "P1",
    )

    assert set(eligible["model"]) == {"a"}
    assert selection.iloc[0]["predictor_accuracy_threshold"] == 0.90
    assert not bool(selection.iloc[0]["threshold_target_met"])


def test_predictor_filter_uses_an_explicit_strict_cutoff():
    from scores import select_predictor_models

    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "subset": "Q1",
            "model": model,
            "accuracy": accuracy,
        }
        for model, accuracy in (("below", 0.94), ("equal", 0.95), ("above", 0.96))
    ])

    eligible, selection = select_predictor_models(
        accuracies,
        accuracies[["irt_model", "fit", "model"]],
        "Q1",
        threshold=0.95,
    )

    assert set(eligible["model"]) == {"above"}
    assert selection.iloc[0]["predictor_accuracy_threshold"] == 0.95


def test_easy_elos_rebuild_components_after_accuracy_filtering():
    from scores import fit_easy_elos

    rounds = pd.DataFrame([
        {"source": "s", "benchmark": "gpqa", "item_id": "i", "vote_id": "s:i",
         "judge": "j", "major": major, "minor": minor,
         "picked_major": 1.0, "n_calls": 1}
        for major, minor in (("a", "b"), ("b", "c"), ("c", "d"))
    ])
    splits = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "half": "easy", "quartile": "Q1",
    }])
    eligible = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "model": model}
        for model in ("a", "c", "d")
    ])

    elos, coverage = fit_easy_elos(
        rounds,
        splits,
        eligible_models=eligible,
    )

    # Removing b breaks the a-b-c bridge. Isolated a has no comparison edge;
    # the retained c-d component is fitted and reported normally.
    assert set(elos["model"]) == {"c", "d"}
    assert set(coverage["component_models"]) == {"c|d"}


def test_gpqa_saturation_fit_excludes_judges_covering_fewer_than_eight_models():
    from scores import fit_saturated_elos

    models = list("abcdefgh")
    rounds = pd.DataFrame([
        {
            "source": "s", "benchmark": "gpqa", "item_id": "i",
            "vote_id": "s:i", "judge": judge, "major": major,
            "minor": minor, "picked_major": 1.0, "n_calls": 1,
        }
        for judge, pairs in (
            ("wide", tuple(zip(models, models[1:]))),
            ("narrow", tuple(zip(models[:4], models[1:4]))),
        )
        for major, minor in pairs
    ])
    splits = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "half": "easy", "tertile": "T1",
        "quartile": "Q1", "quintile": "P1",
    }])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "subset": "easy",
            "model": model, "accuracy": 0.8,
        }
        for model in models
    ])

    elos, coverage = fit_saturated_elos(
        rounds,
        splits,
        accuracies,
        bank="easy",
    )

    assert set(elos["judge"]) == {"wide", "consensus"}
    assert set(coverage["coverage_excluded_judges"]) == {"narrow"}
    assert set(coverage["minimum_judge_models"]) == {8}
    assert set(coverage.query("judge == 'consensus'")["n_judges"]) == {1}


def test_lomo_does_not_train_on_held_out_target():
    from metrics import lomo_predictions

    frame = pd.DataFrame({
        "irt_model": ["fixed_3pl"] * 4,
        "fit": ["gpqa"] * 4,
        "bank": ["easy"] * 4,
        "judge": ["j"] * 4,
        "component": [0] * 4,
        "model": ["a", "b", "c", "held"],
        "elo": [0.0, 1.0, 2.0, 3.0],
        "easy_accuracy": [0.1, 0.2, 0.3, 0.4],
        "hard_accuracy": [0.0, 0.2, 0.4, 1.0],
    })

    held = lomo_predictions(frame)
    held = held[held["model"] == "held"].set_index("method")

    assert held.loc["elo", "prediction"] == 0.6
    assert held.loc["easy_accuracy", "prediction"] == 0.6
    assert held.loc["mean", "prediction"] == 0.2
    assert held.loc["identity", "prediction"] == 0.4


def test_rank_metrics_uses_one_matched_model_set_for_both_predictors():
    from metrics import rank_metrics

    frame = pd.DataFrame({
        "irt_model": ["fixed_3pl"] * 4,
        "fit": ["gpqa"] * 4,
        "bank": ["easy"] * 4,
        "judge": ["j"] * 4,
        "component": [0] * 4,
        "model": ["a", "b", "c", "missing-baseline"],
        "elo": [0.0, 1.0, 2.0, 3.0],
        "easy_accuracy": [0.1, 0.2, 0.3, np.nan],
        "hard_accuracy": [0.0, 0.2, 0.4, 1.0],
    })

    got = rank_metrics(frame)

    assert set(got["method"]) == {"elo", "easy_accuracy"}
    assert set(got["n_models"]) == {3}


def test_kendall_tau_a_keeps_predictor_ties_in_the_denominator():
    from metrics import kendall_tau_a

    # Two pairs are ordered correctly; the tied predictor pair supplies no
    # information but still remains one of the three pairs being evaluated.
    assert kendall_tau_a([0, 0, 1], [0, 1, 2]) == 2 / 3


def test_analysis_frame_can_match_q1_to_later_quartile():
    from scores import analysis_frame

    elos = pd.DataFrame([{
        "irt_model": "fixed_3pl",
        "fit": "gpqa",
        "bank": "Q1",
        "judge": "j",
        "component": 0,
        "model": "m",
        "elo": 1.0,
    }])
    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": "Q1", "model": "m", "accuracy": 0.8},
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": "Q4", "model": "m", "accuracy": 0.2},
    ])

    got = analysis_frame(
        elos,
        accuracies,
        predictor_subset="Q1",
        target_subset="Q4",
    ).iloc[0]

    assert got["easy_accuracy"] == 0.8
    assert got["hard_accuracy"] == 0.2


def test_quartile_analysis_reports_q1_to_each_later_quartile():
    from metrics import quartile_analysis

    models = ["a", "b", "c", "d"]
    elos = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "Q1", "judge": "j", "component": 0, "model": model, "elo": i}
        for i, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "subset": quartile, "model": model, "accuracy": (i + offset) / 10}
        for quartile, offset in (
            ("Q1", 1), ("Q2", 2), ("Q3", 3), ("Q4", 4), ("Q2–Q4", 3),
        )
        for i, model in enumerate(models)
    ])

    got = quartile_analysis(elos, accuracies)

    assert set(got["target_quartile"]) == {"Q2", "Q3", "Q4", "Q2–Q4"}
    assert set(got["metric"]) == {"tau_a", "mae"}
    assert set(got["method"]) == {"elo", "easy_accuracy"}


def test_partition_analysis_returns_model_level_lomo_rows_for_each_target():
    from metrics import partition_analysis

    models = ["a", "b", "c", "d"]
    elos = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "bank": "T1",
            "judge": "consensus",
            "component": "0:0",
            "model": model,
            "elo": float(i),
        }
        for i, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "subset": subset,
            "model": model,
            "accuracy": (i + offset) / 10,
        }
        for subset, offset in (("T1", 1), ("T2", 2), ("T3", 3))
        for i, model in enumerate(models)
    ])

    metrics, predictions = partition_analysis(
        elos,
        accuracies,
        predictor_bin="T1",
        target_bins=("T2", "T3"),
    )

    assert set(metrics["partition"]) == {"tertile"}
    assert set(metrics["target_bin"]) == {"T2", "T3"}
    assert set(metrics["metric"]) == {"tau_a", "mae"}
    assert metrics[["ci_low", "ci_high"]].notna().all().all()
    assert set(metrics["ci_method"].dropna()) == {
        "paired model bootstrap percentile 95%",
    }
    assert set(predictions["target_bin"]) == {"T2", "T3"}
    assert set(predictions["method"]) == {"elo", "easy_accuracy", "mean", "identity"}
    elo_rows = predictions[predictions["method"] == "elo"]
    assert dict(zip(elo_rows["model"], elo_rows["predictor"])) == {
        "a": 0.0,
        "b": 1.0,
        "c": 2.0,
        "d": 3.0,
    }


def test_partition_analysis_bootstraps_models_for_each_judge():
    from metrics import partition_analysis

    models = list("abcde")
    elos = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
            "judge": judge, "component": "0:0", "model": model,
            "elo": float(index),
        }
        for judge in ("j1", "j2", "consensus")
        for index, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "subset": subset,
            "model": model, "accuracy": (index + offset) / 10,
        }
        for subset, offset in (("easy", 1), ("hard", 2))
        for index, model in enumerate(models)
    ])

    metrics, _ = partition_analysis(
        elos,
        accuracies,
        predictor_bin="easy",
        target_bins=("hard",),
    )

    intervals = metrics[metrics["method"].isin(["elo", "easy_accuracy"])]
    assert set(intervals["judge"]) == {"j1", "j2", "consensus"}
    assert intervals[["ci_low", "ci_high"]].notna().all().all()
    assert set(intervals["ci_method"]) == {
        "paired model bootstrap percentile 95%",
    }


def test_randomized_nulls_coin_flip_votes_and_permute_accuracy():
    from metrics import partition_analysis
    from nulls import randomized_null_metrics
    from scores import fit_easy_elos

    models = list("abcde")
    rounds = pd.DataFrame([
        {
            "source": "s", "benchmark": "gpqa", "item_id": "i",
            "vote_id": "s:i", "judge": "j", "major": major,
            "minor": minor, "picked_major": 1.0, "n_calls": 2,
        }
        for major, minor in zip(models, models[1:])
    ])
    splits = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "half": "easy", "tertile": "T1",
        "quartile": "Q1", "quintile": "P1",
    }])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "subset": subset,
            "model": model, "accuracy": value,
        }
        for subset, values in (
            ("easy", (0.6, 0.7, 0.8, 0.9, 1.0)),
            ("hard", (0.2, 0.4, 0.5, 0.7, 0.8)),
        )
        for model, value in zip(models, values)
    ])
    elos, _ = fit_easy_elos(rounds, splits)
    _, predictions = partition_analysis(
        elos,
        accuracies,
        predictor_bin="easy",
        target_bins=("hard",),
    )

    got = randomized_null_metrics(
        rounds,
        splits,
        elos,
        predictions,
        resamples=20,
        seed=4,
    )

    assert set(got["method"]) == {"elo", "easy_accuracy"}
    assert set(got["null_type"]) == {
        "coin-flip judgments",
        "permuted easy accuracy",
    }
    assert set(got["metric"]) == {"tau_a", "mae"}
    assert set(got["n_randomizations"]) == {20}
    assert got[["null_median", "ci_low", "ci_high", "p_value"]].notna().all().all()
    assert got["p_value"].between(0, 1).all()


def test_analytic_nulls_use_tau_zero_and_intercept_only_lomo():
    from metrics import partition_analysis
    from nulls import analytic_null_metrics

    models = list("abcde")
    elos = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy",
            "judge": "consensus", "component": "0:0", "model": model,
            "elo": float(index),
        }
        for index, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "subset": subset,
            "model": model, "accuracy": value,
        }
        for subset, values in (
            ("easy", (0.6, 0.7, 0.8, 0.9, 1.0)),
            ("hard", (0.2, 0.4, 0.5, 0.7, 0.8)),
        )
        for model, value in zip(models, values)
    ])
    _, predictions = partition_analysis(
        elos, accuracies, predictor_bin="easy", target_bins=("hard",),
    )

    got = analytic_null_metrics(predictions)

    assert set(got["method"]) == {"elo", "easy_accuracy"}
    assert set(got["metric"]) == {"tau_a", "mae"}
    assert got.loc[got["metric"] == "tau_a", "null_median"].eq(0).all()
    assert set(got.loc[got["metric"] == "mae", "null_type"]) == {
        "intercept-only LOMO",
    }
    assert set(got["n_randomizations"]) == {0}
    assert got["p_value"].between(0, 1).all()
    assert "pooled" in set(got["component"])


def test_partition_analysis_supports_the_half_split():
    from metrics import partition_analysis

    models = ["a", "b", "c", "d"]
    elos = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "bank": "easy",
            "judge": "consensus",
            "component": "0:0",
            "model": model,
            "elo": float(i),
        }
        for i, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "subset": subset,
            "model": model,
            "accuracy": (i + offset) / 10,
        }
        for subset, offset in (("easy", 1), ("hard", 2))
        for i, model in enumerate(models)
    ])

    metrics, predictions = partition_analysis(
        elos,
        accuracies,
        predictor_bin="easy",
        target_bins=("hard",),
    )

    assert set(metrics["partition"]) == {"halves"}
    assert set(metrics["target_bin"]) == {"hard"}
    assert set(predictions["predictor_bin"]) == {"easy"}


def test_model_bootstrap_intervals_resample_paired_models():
    from metrics import model_bootstrap_intervals

    predictions = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "bank": "Q1",
            "partition": "quartile",
            "predictor_bin": "Q1",
            "target_bin": "Q4",
            "judge": "consensus",
            "component": "0:0",
            "model": model,
            "method": method,
            "predictor": value,
            "prediction": value,
            "observed": value,
        }
        for method in ("elo", "easy_accuracy")
        for model, value in zip("abcde", np.linspace(0.1, 0.9, 5))
    ])

    got = model_bootstrap_intervals(predictions, n_resamples=200, seed=3)

    tau = got[got["metric"] == "tau_a"]
    mae = got[got["metric"] == "mae"]
    assert np.allclose(tau[["ci_low", "ci_high"]], 1.0)
    assert (mae[["ci_low", "ci_high"]] == 0.0).all().all()
    assert set(got["n_bootstrap_models"]) == {5}


def test_paired_method_randomization_compares_the_same_models():
    from metrics import paired_method_randomization

    observed = np.linspace(0.1, 0.9, 8)
    predictions = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "bank": "Q1",
            "partition": "quartile",
            "predictor_bin": "Q1",
            "target_bin": "Q4",
            "judge": "consensus",
            "component": "0:0",
            "model": model,
            "method": method,
            "predictor": value if method == "elo" else -value,
            "prediction": value if method == "elo" else 1 - value,
            "observed": value,
        }
        for method in ("elo", "easy_accuracy")
        for model, value in zip("abcdefgh", observed)
    ])

    got = paired_method_randomization(predictions, n_resamples=999, seed=3)

    assert set(got["metric"]) == {"tau_a", "mae"}
    assert set(got["n_comparison_models"]) == {8}
    assert set(got["comparison_test"]) == {
        "paired model-label randomization, two-sided",
    }
    assert got["comparison_p_value"].lt(0.05).all()


def test_cross_benchmark_analysis_keeps_both_directions():
    from metrics import cross_benchmark_analysis

    models = ["a", "b", "c", "d"]
    elos = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": fit, "bank": "easy", "judge": "j", "component": 0, "model": model, "elo": i}
        for fit in ("gpqa", "aime")
        for i, model in enumerate(models)
    ])
    accuracies = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": fit, "subset": subset, "model": model, "accuracy": (i + offset) / 10}
        for fit, offset in (("gpqa", 1), ("aime", 2))
        for subset in ("easy", "hard")
        for i, model in enumerate(models)
    ])

    got = cross_benchmark_analysis(elos, accuracies)

    assert set(got["direction"]) == {"gpqa_to_aime", "aime_to_gpqa"}
    assert set(got["metric"]) == {"tau_a", "mae"}


def test_newer_stronger_analysis_counts_each_exclusion_reason():
    from metrics import newer_stronger_analysis

    frame = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "bank": "easy", "judge": "j", "component": 0, "model": model, "elo": i, "easy_accuracy": 0.5, "hard_accuracy": 0.4}
        for i, model in enumerate(("keep", "newer-only", "stronger-only", "missing-date"))
    ])
    model_info = pd.DataFrame([
        {"fit": "gpqa", "model": "j", "release": "2025-01", "full_accuracy": 0.5},
        {"fit": "gpqa", "model": "keep", "release": "2025-02", "full_accuracy": 0.6},
        {"fit": "gpqa", "model": "newer-only", "release": "2025-02", "full_accuracy": 0.4},
        {"fit": "gpqa", "model": "stronger-only", "release": "2024-12", "full_accuracy": 0.6},
        {"fit": "gpqa", "model": "missing-date", "release": None, "full_accuracy": 0.7},
    ])

    got = newer_stronger_analysis(frame, model_info)

    assert set(got["included_models"]) == {"keep"}
    assert set(got["n_candidates_before"]) == {4}
    assert set(got["n_candidates_after"]) == {1}
    assert set(got["excluded_missing_release"]) == {1}
    assert set(got["excluded_not_newer"]) == {1}
    assert set(got["excluded_not_stronger"]) == {1}
