import numpy as np
import pandas as pd

from protocols.pairwise.linear import weighted_theta


def test_weighted_theta_recovers_targets_and_downweights_a_noisy_bench():
    """Two benches: one whose logits follow k*dtheta, one that is pure noise. Targets come back near theta*."""
    rng = np.random.default_rng(0)
    theta = {"r1": -1.5, "r2": -0.5, "r3": 0.5, "r4": 1.5, "t1": -0.5, "t2": 1.0}
    refs = pd.Series({m: theta[m] for m in ["r1", "r2", "r3", "r4"]})
    rows = []
    for a in theta:
        for b in theta:
            if a >= b:
                continue
            for bench, k, noise in [("good", 1.0, 0.0), ("style", 0.0, 3.0)]:
                p = 1 / (1 + np.exp(-(k * (theta[a] - theta[b]) + rng.normal(0, noise))))
                for item in range(60):
                    rows.append((bench, "judge", item, a, b, float(rng.random() < p)))
    verdicts = pd.DataFrame(rows, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    got = weighted_theta(verdicts, refs, ["t1", "t2"])
    assert abs(got["t1"] - theta["t1"]) < 0.4 and abs(got["t2"] - theta["t2"]) < 0.4
