import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae_describe import (describe_features, feature_group, fidelity, interpretation_examples,
                                     parse_description, parse_presence, scoring_examples,
                                     split_shared_by_group, truncate_words)

SETTINGS = {"interpreter_model": "fake", "annotator_model": "fake", "interpreter_max_tokens": 10,
            "annotator_max_tokens": 10, "n_candidates": 2, "interpret_examples": 3,
            "contrast_examples": 2, "top_percentile": 50, "scoring_examples": 20, "max_words": 40,
            "annotation_batch_size": 7,
            "min_correlation": 0.0, "p_value_threshold": None, "min_prevalence": 0.05, "seed": 0}


def group_rows(n, items=10):
    return pd.DataFrame({"transcript_key": [f"k{i}" for i in range(n)], "benchmark": ["gpqa"] * n,
                         "item_group_key": [f"item{i % items}" for i in range(n)],
                         "transcript": [f"transcript {i} " + "x " * 100 for i in range(n)]})


def test_example_sampling_and_parsing():
    activations = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -1, -2] + [0] * 8, dtype=float)
    rows = group_rows(20)                                    # items repeat every 10 rows
    chosen = interpretation_examples(activations, rows, n_firing=3, n_zero=2, top_percentile=50, seed=0)
    assert set(chosen["firing"]) <= {0, 1, 2, 3, 4} and len(chosen["firing"]) == 3
    assert all(activations[chosen["zero"]] == 0) and len(chosen["zero"]) == 2
    items = rows["item_group_key"].to_numpy()
    assert set(items[chosen["zero"]]) <= set(items[chosen["firing"]])   # contrasts come from the firing items
    short_bin = interpretation_examples(activations, rows, n_firing=4, n_zero=1, top_percentile=99, seed=0)
    assert sorted(short_bin["firing"]) == [0, 1, 2, 3]      # bin too small -> walk down by activation
    one_item = interpretation_examples(activations, group_rows(20, items=1), n_firing=3, n_zero=2,
                                       top_percentile=50, seed=0)
    assert len(one_item["firing"]) == 1                     # at most one firing row per item
    scored = scoring_examples(activations, 20, 0)
    assert len(scored) == 12 and all(activations[scored] != 0)
    assert parse_description('- "uses numbered steps"') == "uses numbered steps"
    matrix = parse_presence('Sure: {"1": [2, 3], "2": []}', 2, 3)
    assert matrix.tolist() == [[0, 1, 1], [0, 0, 0]] and parse_presence("no json", 2, 3) is None
    assert truncate_words("a b c d e f", 4) == "a b\n[...middle of transcript truncated...]\ne f"


def test_fidelity_is_signed_pearson_with_p_value():
    activations = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    correlation, p_value = fidelity(np.array([1, 1, 1, 0, 0, 0], dtype=float), activations)
    assert correlation > 0.9 and p_value < 0.05
    assert fidelity(np.array([1, 1, 1, 1, 1, 1], dtype=float), activations) == (0.0, 1.0)


def test_describe_features_keeps_the_best_faithful_candidate():
    rows = group_rows(80)
    activations = np.zeros((80, 2))
    activations[:20, 0] = np.arange(20, 0, -1)          # feature 0: fires on the first 20 rows
    activations[20:30, 0] = -1.0                        # ... and negatively on ten more
    activations[:3, 1] = [3, 2, 1]                      # feature 1: too rare for prevalence
    activations[3:80, 1] = 0.0

    interpretations = iter(['- "long answer"', '- "short answer"', '- "rare thing"', '- "rare thing"'])

    def ask(model, prompt, max_tokens):
        if prompt.startswith("# Task Overview"):
            return next(interpretations)
        # batched annotator: concept "long answer" is present exactly on the positively firing rows
        concepts = re.findall(r'^\d+\. "(.*)"$', prompt.split("CONCEPTS:\n")[1].split("\n\nTRANSCRIPT")[0], re.M)
        numbers = [int(m) for m in re.findall(r"TRANSCRIPT \d+:\ntranscript (\d+) ", prompt)]
        present = {}
        for index, concept in enumerate(concepts, 1):
            present[str(index)] = [n + 1 for n, row in enumerate(numbers)
                                   if (concept == "long answer" and row < 20) or (concept == "short answer" and row >= 20 and row % 2)]
        return json.dumps(present)

    table = describe_features(activations, ["gpqa:0", "gpqa:1"], rows, {"gpqa": ["gpqa"]},
                              SETTINGS, ask, width=2, workers=1)
    first = table.iloc[0]
    assert first.feature == "gpqa:0" and first.kept and first.description == "long answer"
    assert first.correlation > 0.5 and first.p_value < 0.025 and first.n_candidates == 2
    second = table.iloc[1]
    assert not second.kept and second.reason == "low_prevalence" and second.description == "rare thing"


def test_offline_mode_returns_none_while_prompts_are_pending():
    rows = group_rows(40)
    activations = np.zeros((40, 1))
    activations[:10, 0] = np.arange(10, 0, -1)

    class Offline:
        def __init__(self):
            self.pending = []

        def __call__(self, model, prompt, max_tokens):
            self.pending.append({"model": model, "prompt": prompt})
            return None

    ask = Offline()
    assert describe_features(activations, ["gpqa:0"], rows, {"gpqa": ["gpqa"]}, SETTINGS, ask, width=1, workers=1) is None
    assert len(ask.pending) == SETTINGS["n_candidates"] and all(item["model"] == "fake" for item in ask.pending)


def test_shared_features_split_into_one_column_per_group():
    activations = np.array([[1.0, 2.0], [3.0, 4.0]])
    split, labels = split_shared_by_group(activations, ["gpqa:0", "shared:0"], "shared", ["gpqa", "aime"])
    assert labels == ["gpqa:0", "shared:0@gpqa", "shared:0@aime"]
    assert split[:, 1].tolist() == split[:, 2].tolist() == [2.0, 4.0]
    assert feature_group("shared:0@aime") == "aime" and feature_group("gpqa:3") == "gpqa"


def test_group_overrides_change_window_and_batch_for_that_group_only():
    rows = pd.concat([group_rows(40), group_rows(40).assign(benchmark="swe", transcript_key=lambda f: "s" + f.transcript_key)],
                     ignore_index=True)
    activations = np.zeros((80, 2))
    activations[:10, 0] = np.arange(10, 0, -1)          # gpqa feature
    activations[40:50, 1] = np.arange(10, 0, -1)        # swe feature
    seen = {"gpqa": [], "swe": []}

    def ask(model, prompt, max_tokens):
        if prompt.startswith("# Task Overview"):
            return '- "thing"'
        group = "swe" if "transcript 0 " not in prompt and "TRANSCRIPT" in prompt and "x x x x x x x x x x x x" not in prompt else "gpqa"
        seen[group].append(prompt.count("TRANSCRIPT "))
        return json.dumps({"1": [1], "2": [1]})

    settings = SETTINGS | {"annotation_batch_size": 10, "max_words": 200,
                           "group_overrides": {"swe": {"annotation_batch_size": 3, "max_words": 4}}}
    describe_features(activations, ["gpqa:0", "swe:0"], rows, {"gpqa": ["gpqa"], "swe": ["swe"]}, settings, ask, width=2, workers=1)
    assert max(seen["gpqa"]) == 10 and max(seen["swe"]) == 3     # swe batches of 3 (truncated to 4 words), gpqa batches of 10
