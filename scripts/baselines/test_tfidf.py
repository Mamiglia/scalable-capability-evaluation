import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfidf import fit_predict, report, run_design  # noqa: E402


def synthetic(seed=0, n_items=40):
    """Two benchmarks, six models; stronger models say "therefore", weaker ones say "maybe"."""
    rng = np.random.default_rng(seed)
    models = [f"m{i}" for i in range(6)]
    transcripts, pairs = [], []
    for bench in ("aime2024", "gpqa"):
        for item in range(n_items):
            for k, model in enumerate(models):
                filler = " ".join(rng.choice(["the", "answer", "is", "we", "compute", "so"], 30))
                cue = "therefore " * (k + 1) + "maybe " * (6 - k)
                transcripts.append(dict(bench=bench, model=model, item=f"i{item}", epoch=0, log="l", text=f"{filler} {cue}"))
            for a in models:
                for b in models:
                    if a != b:
                        pairs.append(dict(bench=bench, item=f"i{item}", model_left=a, model_right=b, epoch_left=0,
                                          epoch_right=0, log_left="l", log_right="l", left_stronger=int(a > b)))
    return pd.DataFrame(pairs), pd.DataFrame(transcripts)


def test_antisymmetric_and_learns_the_cue():
    pairs, transcripts = synthetic()
    from tfidf import transcript_rows
    left, right = transcript_rows(pairs, transcripts, "left"), transcript_rows(pairs, transcripts, "right")
    predictions = run_design("item-fold-0", pairs, transcripts, left, right)
    assert report(predictions).accuracy.iloc[0] > 0.9
    flipped = predictions.merge(predictions, left_on=["bench", "item", "model_left", "model_right"],
                                right_on=["bench", "item", "model_right", "model_left"])
    assert np.allclose(flipped.predicted_x + flipped.predicted_y, 1, atol=1e-6)


def test_fit_predict_is_odd_in_the_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = (x[:, 0] > 0).astype(int)
    p_plus, _ = fit_predict(x, y, np.arange(200) // 4, x[:10])
    p_minus, _ = fit_predict(x, y, np.arange(200) // 4, -x[:10])
    assert np.allclose(p_plus + p_minus, 1, atol=1e-6)
