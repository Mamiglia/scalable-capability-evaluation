"""Twin scorer and baselines for the score model.

TwinScorer: theta_j(t) = MLP(embedding(t), judge_embedding_j); verdict
logit = (theta_1 - theta_2) / T_j + b_j. The same MLP scores both slots,
so swapping transcripts exactly flips the content term; position bias
lives only in b_j.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn


class TwinScorer(nn.Module):
    def __init__(self, embed_dim: int, n_judges: int, judge_embed_dim: int,
                 hidden_dims: list[int], dropout: float,
                 judge_embed_init_std: float):
        super().__init__()
        self.judge_embedding = nn.Embedding(n_judges, judge_embed_dim)
        # Keep the judge features on the scale of the unit-norm embedding dims.
        nn.init.normal_(self.judge_embedding.weight, std=judge_embed_init_std)
        layers, width = [], embed_dim + judge_embed_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        final = nn.Linear(width, 1)
        # theta starts at 0: the model starts from the position baseline
        # (logit = b_j) and learns content on top.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.mlp = nn.Sequential(*layers, final)
        # Per-judge position bias and log-temperature (T_j = exp(tau_j)).
        self.b = nn.Parameter(torch.zeros(n_judges))
        self.tau = nn.Parameter(torch.zeros(n_judges))

    def theta(self, e: torch.Tensor, judge_idx: torch.Tensor) -> torch.Tensor:
        """Per-transcript score for the given judge."""
        judge = self.judge_embedding(judge_idx)
        return self.mlp(torch.cat([e, judge], dim=-1)).squeeze(-1)

    def forward(self, e1, e2, judge_idx):
        """Verdict logit for P(pick slot 1)."""
        tau = self.tau - self.tau.mean()  # scale gauge: mean log-temperature 0
        temperature = torch.exp(tau)[judge_idx]
        # Same network scores both slots; only the difference carries content.
        content = self.theta(e1, judge_idx) - self.theta(e2, judge_idx)
        return content / temperature + self.b[judge_idx]

    @torch.no_grad()
    def score_unseen_judge(self, e1, e2):
        """Held-out-judge rule: mean judge embedding, b = 0, mean T."""
        mean_judge = self.judge_embedding.weight.mean(0, keepdim=True)

        def theta(e):
            return self.mlp(torch.cat([e, mean_judge.expand(len(e), -1)],
                                      dim=-1)).squeeze(-1)

        tau = self.tau - self.tau.mean()
        return (theta(e1) - theta(e2)) / torch.exp(tau).mean()


def symmetry_check(seed: int = 0, n_rows: int = 8, embed_dim: int = 32,
                   n_judges: int = 3):
    """logit(e1,e2) + logit(e2,e1) must equal 2*b_j exactly: swapping
    transcripts flips the content term and leaves position bias alone."""
    torch.manual_seed(seed)
    model = TwinScorer(embed_dim, n_judges, judge_embed_dim=2,
                       hidden_dims=[16], dropout=0.0,
                       judge_embed_init_std=1.0).eval()
    with torch.no_grad():
        model.b.uniform_(-1, 1)
        # Undo the zero final-layer init so the content term is nonzero
        # and the check is not vacuous.
        model.mlp[-1].weight.normal_()
        model.mlp[-1].bias.normal_()
        e1, e2 = torch.randn(n_rows, embed_dim), torch.randn(n_rows, embed_dim)
        judge_idx = torch.randint(0, n_judges, (n_rows,))
        residual = (model(e1, e2, judge_idx) + model(e2, e1, judge_idx)
                    - 2 * model.b[judge_idx])
    assert residual.abs().max() < 1e-5, f"symmetry broken: {residual.abs().max()}"


def fit_position(train_rows: pd.DataFrame) -> dict[str, float]:
    """Per-judge b as the logit of the train slot-1 rate (the ML fit of
    P(y=1) = sigma(b_j))."""
    rates = train_rows.groupby("judge")["y"].mean().clip(1e-3, 1 - 1e-3)
    return np.log(rates / (1 - rates)).to_dict()


def predict_position(b_by_judge: dict, rows: pd.DataFrame) -> np.ndarray:
    """P(pick slot 1); unseen judges get b = 0."""
    logits = rows["judge"].map(b_by_judge).fillna(0.0).to_numpy()
    return 1 / (1 + np.exp(-logits))


def fit_dlogit(train_rows: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per judge: P(y=1) = sigma(s_j * dlogit + b_j), maximum likelihood.
    dlogit is the capability-gap column built by the caller."""
    from sklearn.linear_model import LogisticRegression
    params = {}
    for judge, judge_rows in train_rows.groupby("judge"):
        fitted = LogisticRegression(C=1e6, max_iter=1000).fit(
            judge_rows[["dlogit"]].to_numpy(), judge_rows["y"])
        params[judge] = (float(fitted.coef_[0, 0]), float(fitted.intercept_[0]))
    return params


def predict_dlogit(params: dict, rows: pd.DataFrame) -> np.ndarray:
    """P(pick slot 1); unseen judges get the mean slope and b = 0."""
    mean_slope = float(np.mean([slope for slope, _ in params.values()]))
    logits = np.array([params[j][0] * d + params[j][1] if j in params
                       else mean_slope * d
                       for j, d in zip(rows["judge"], rows["dlogit"])])
    return 1 / (1 + np.exp(-logits))