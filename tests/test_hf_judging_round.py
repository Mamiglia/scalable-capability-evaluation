import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "analyses"))

try:
    import hf_judging_round
except ImportError:
    hf_judging_round = None

summarize = getattr(hf_judging_round, "summarize", None)
orient_accuracy = getattr(hf_judging_round, "orient_accuracy", None)
load_benchmark_scores = getattr(hf_judging_round, "load_benchmark_scores", None)
keep_judge = getattr(hf_judging_round, "keep_judge", None)


def test_summary_averages_judges_not_verdicts():
    assert summarize is not None, "hf_judging_round.summarize is not implemented"
    rows = [
        {"eval": "GPQA", "judge": "j1", "pair": "a", "gap": 1, "correct": 45, "n": 50},
        {"eval": "GPQA", "judge": "j1", "pair": "a", "gap": 1, "correct": 45, "n": 50},
        {"eval": "GPQA", "judge": "j2", "pair": "a", "gap": 1, "correct": 1, "n": 1},
        {"eval": "GPQA", "judge": "j3", "pair": "a", "gap": 1, "correct": 1, "n": 1},
    ]

    summary, per_judge = summarize(rows, [0, 5])

    assert [r["accuracy"] for r in per_judge] == pytest.approx([0.9, 1, 1])
    assert summary[0]["accuracy"] == pytest.approx((0.9 + 1 + 1) / 3)
    assert summary[0]["n_judges"] == 3
    assert summary[0]["n_verdicts"] == 102


def test_accuracy_is_oriented_to_the_stronger_model():
    assert orient_accuracy is not None, "hf_judging_round.orient_accuracy is not implemented"
    assert orient_accuracy(0.8, 0.8, 0.6) == pytest.approx(0.8)
    assert orient_accuracy(0.8, 0.6, 0.8) == pytest.approx(0.2)


def test_benchmark_scores_pool_epochs_by_sample_count(tmp_path):
    assert load_benchmark_scores is not None, "load_benchmark_scores is not implemented"
    for epoch, accuracy, n in ((0, 0.5, 10), (1, 1.0, 30)):
        path = tmp_path / "gpqa" / "hash" / "model-a" / str(epoch) / "logs.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"run.eval": {
            "status": "success",
            "invalidated": False,
            "results": {"scores": [{
                "scored_samples": n,
                "metrics": {"accuracy": {"value": accuracy}},
            }]},
        }}))

    assert load_benchmark_scores(tmp_path)[("GPQA", "model-a")] == pytest.approx(0.875)


def test_phi_is_never_in_the_judge_pool():
    assert keep_judge is not None, "keep_judge is not implemented"
    assert not keep_judge("phi3-medium")
    assert not keep_judge("Phi-3-medium-128k-instruct")
    assert keep_judge("qwen3-8b")
