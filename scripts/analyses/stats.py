"""Statistical helpers for the judge analysis, ported from the lasr pilot.

The unit of analysis is the judged pair (item within a run): its two A/B
orders average to a score of 0, 0.5 or 1, which cancels position bias.
"""

import math

import numpy as np
import pandas as pd

# Slot labels are "1"/"2", which read_csv would otherwise type as integers and
# silently fail every comparison against the string labels.
SLOT_COLUMNS = ["picked", "slot_of_major"]

# Some models answer an MCQ item with the ANSWER line and nothing else
# Stripping that line leaves ~12 characters, so the judge is shown no reasoning at all
# Over the pilot transcripts, degenerate ones are ~12 characters and the next shortest real one is 234
MIN_REASONING_CHARS = 100


def drop_degenerate(verdicts):
    """Drop items where either transcript is just the answer with no reasoning.

    Filters on the pre-truncation length, so full and truncated runs keep the
    same item set and stay comparable. Returns (kept, n_dropped).
    """
    cols = ["reasoning_len_major", "reasoning_len_minor"]
    if not all(c in verdicts for c in cols):
        return verdicts, 0
    ok = verdicts[cols].min(axis=1) >= MIN_REASONING_CHARS
    return verdicts[ok], int((~ok).sum())


def load_verdicts(path):
    """Read verdicts.csv with slot labels as strings, degenerate items dropped.

    Returns (verdicts, n_dropped). Read without the dtype fix, "1"/"2" arrive as
    integers and every slot comparison quietly returns nothing.
    """
    df = pd.read_csv(path, dtype={c: str for c in SLOT_COLUMNS})
    return drop_degenerate(df)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well behaved at small n and extreme rates."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def counterbalanced(run_df, value_col: str = "picked_major") -> np.ndarray:
    """Per-item scores for one run, averaged over its parsed A/B orders.

    Rows with NaN in value_col (unparsed verdicts) are dropped first; an item
    with one parsed order keeps that order's score. Every item counts once.
    """
    parsed = run_df.dropna(subset=[value_col])
    return parsed.groupby("item")[value_col].mean().to_numpy()


def sem_ci(scores) -> tuple[float, float, float]:
    """(mean, lo, hi) with lo/hi = mean +/- 1.96 * SEM over pairs.

    The unit is the pair, whose counterbalanced score is 0, 0.5 or 1. Wilson
    assumes Bernoulli scores and does not apply here.
    """
    if len(scores) == 0:
        return 0.0, 0.0, 1.0
    a = np.asarray(scores, float)
    m = float(a.mean())
    sem = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def perm_p(scores, n_perm: int = 20000, seed: int = 0) -> float | None:
    """Two-sided permutation p-value against "the judge has no capability signal".

    The null shuffles which of the two compared models is labelled the
    stronger one; swapping that label turns a pair's score s into 1 - s, so
    the null flips each pair independently. Reported against 0.5, two-sided.
    """
    if len(scores) < 2:
        return None
    a = np.asarray(scores, float)
    obs = abs(a.mean() - 0.5)
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, (n_perm, len(a))).astype(bool)
    null = np.where(flips, 1.0 - a, a).mean(axis=1)
    return float((np.abs(null - 0.5) >= obs - 1e-12).mean())


def stars(p: float | None) -> str:
    """Significance marker: *** <.001, ** <.01, * <.05, else n.s."""
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def sigmoid_fit(gaps, scores) -> tuple[float, float, float, float] | None:
    """Fit score = sigmoid(a + d * gap) by IRLS. Returns (a, d, se_a, se_d).

    sigmoid(a) is the extrapolated accuracy at zero capability gap, where no
    capability difference remains to read; anything above chance there is the
    judge identifying which model wrote the transcript. Compare with the
    self-pair control:

        self-pair ~ 0.5, sigmoid(a) ~ 0.5   -> signal is capability-driven
        self-pair ~ 0.5, sigmoid(a) >> 0.5  -> fingerprinting

    Self-pairs are excluded from the fit: they are the control the fit is
    judged against, and their y value is slot preference.
    """
    g = np.asarray(gaps, float)
    y = np.asarray(scores, float)
    if len(g) < 10 or len(set(g.tolist())) < 2:
        return None
    X = np.column_stack([np.ones_like(g), g])
    beta = np.zeros(2)
    for _ in range(200):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-6, None)
        try:
            step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(2),
                                   X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        beta += np.clip(step, -2, 2)
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    W = np.clip(p * (1 - p), 1e-6, None)
    try:
        cov = np.linalg.inv(X.T @ (W[:, None] * X))
        se_a, se_d = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se_a = se_d = np.nan
    return float(beta[0]), float(beta[1]), float(se_a), float(se_d)
