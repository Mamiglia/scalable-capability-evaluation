"""Mean direct ratings -> capability, linearly calibrated on the training models' theta*.

One row per judgment: bench, model, item, score (a number from 1 to 10, decimals preserved).
Repeated judging epochs are averaged per transcript before taking the model mean.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def mean_scores(scores: pd.DataFrame) -> pd.Series:
    """One mean per model, weighting each transcript equally; invalid ratings stay missing.

    Retried copies must already be deduplicated (from_logs does this). Distinct judging
    epochs of one transcript contribute to its mean, not extra weight among transcripts.
    """
    values = pd.to_numeric(scores.score, errors="coerce")
    valid = values.between(1, 10) # hard coded in - make sure prompt matches
    # this!
    transcripts = scores.assign(score=values.where(valid)).groupby(["model", "bench", "item"]).score.mean()
    return transcripts.groupby("model").mean()


def calibrate(scores: pd.DataFrame, anchors: pd.Series) -> pd.Series:
    """theta-hat indexed by model: OLS of training theta* on their mean ratings.

    Each training model (reference or historical) contributes one row. Missing ratings or
    theta* are excluded; fewer than two usable models leaves every forecast NaN. Constant
    training ratings produce a zero slope and the mean training theta*. Forecasts are not clipped.
    """
    means = mean_scores(scores)
    reference = means.reindex(anchors.index)
    known = np.isfinite(reference) & np.isfinite(anchors)
    prediction = pd.Series(np.nan, index=means.index)
    if known.sum() < 2:
        return prediction
    estimator = LinearRegression().fit(reference[known].to_frame(), anchors[known])
    observed = np.isfinite(means)
    prediction.loc[observed] = estimator.predict(means[observed].to_frame())
    return prediction
