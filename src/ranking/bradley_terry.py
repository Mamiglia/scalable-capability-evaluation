"""
Bradley-Terry (ELO) abilities from pairwise verdicts.

One row per verdict: model_left, model_right, pick_left (0/1, or a probability).
P(left picked) = sigmoid(ability_left - ability_right). No position term; a pair
whose two shown orders disagree contributes one win each way, i.e. a tie.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

GROUP = ["bench", "judge", "item"]  # one group = one pair judged (in both orders) by one judge on one item
RIDGE = 0.01  # penalty on ability^2; keeps an unbeaten model finite


def pair_groups(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Per verdict: its (bench, judge, item, unordered pair) key and whether that group's shown orders disagree."""
    pair = np.sort(verdicts[["model_left", "model_right"]].to_numpy(), axis=1)  # unordered pair, alphabetical
    keys = verdicts[GROUP].assign(pair_a=pair[:, 0], pair_b=pair[:, 1])
    # pick reoriented to "picked pair_a", so both shown orders are comparable
    pick_a = np.where(verdicts.model_left.to_numpy() == pair[:, 0],
                      verdicts.pick_left.to_numpy(float), 1 - verdicts.pick_left.to_numpy(float))
    by_group = pd.Series(pick_a, index=verdicts.index).groupby([keys[c] for c in keys], sort=False)
    return keys.assign(tied=by_group.transform("max") > by_group.transform("min"))  # orders disagree


def fit_abilities(left: np.ndarray, right: np.ndarray, pick_left: np.ndarray, n_models: int,
                  ridge: float = RIDGE) -> np.ndarray:
    """Maximum-likelihood abilities, centred to mean zero. left/right are model indices per verdict."""

    def loss(theta):
        logit = theta[left] - theta[right]
        # logistic log-loss with fractional targets, plus ridge
        return np.sum(pick_left * np.logaddexp(0, -logit) + (1 - pick_left) * np.logaddexp(0, logit)) + ridge * np.sum(theta ** 2)

    theta = minimize(loss, np.zeros(n_models), method="L-BFGS-B").x
    return theta - theta.mean()  # only differences are identified


def rank(verdicts: pd.DataFrame, drop_ties: bool = False, ridge: float = RIDGE) -> pd.DataFrame:
    """
    Rank the models in a verdict table.

    Columns needed: bench, judge, item, model_left, model_right, pick_left.
    drop_ties removes every group whose shown orders disagree.
    Returns one row per model: ability, rank (1 = strongest), n_comparisons, n_wins, n_ties.
    """
    groups = pair_groups(verdicts)
    if drop_ties:
        verdicts, groups = verdicts[~groups.tied], groups[~groups.tied]

    # fit on integer-indexed models
    models = sorted(set(verdicts.model_left) | set(verdicts.model_right))
    index = {m: i for i, m in enumerate(models)}
    left = verdicts.model_left.map(index).to_numpy()
    right = verdicts.model_right.map(index).to_numpy()
    pick_left = verdicts.pick_left.to_numpy(float)
    ability = fit_abilities(left, right, pick_left, len(models), ridge)

    # counts per model: each verdict counts once for each side
    per_side = pd.concat([pd.DataFrame({"model": verdicts.model_left, "win": pick_left}),
                          pd.DataFrame({"model": verdicts.model_right, "win": 1 - pick_left})])
    counts = per_side.groupby("model").win.agg(n_comparisons="size", n_wins="sum")
    tied_pairs = groups[groups.tied].drop_duplicates()  # one row per tied group
    n_ties = pd.concat([tied_pairs.pair_a, tied_pairs.pair_b]).value_counts().rename("n_ties")

    out = pd.DataFrame({"model": models, "ability": ability}).join(counts, on="model").join(n_ties, on="model")
    out["n_ties"] = out.n_ties.fillna(0).astype(int)
    out["rank"] = out.ability.rank(ascending=False, method="min").astype(int)
    return out[["model", "ability", "rank", "n_comparisons", "n_wins", "n_ties"]].sort_values("rank").reset_index(drop=True)
