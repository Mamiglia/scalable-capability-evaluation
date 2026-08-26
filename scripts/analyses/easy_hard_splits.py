#!/usr/bin/env python3
"""Easy/hard splits of the pooled GPQA + HLE-gold set, built two ways and compared.

Builders: IRT = pooled-2PL difficulty b (analysis/irt/pooled_gold_difficulties.csv);
P = mean correctness over models (no model). Splits: easiest 25% vs hardest 25%;
easiest 25% vs rest; 50/50; easy GPQA (easiest 25% of GPQA) vs all HLE-gold; all GPQA vs HLE-gold.
For each: overlap of the two builders' easy and hard sets, and per builder the mean model
accuracy on each set and the easy-vs-hard correlation across models (Pearson r, Spearman rho).
Run:  uv run python scripts/analyses/easy_hard_splits.py
"""
import sys
from pathlib import Path

import pandas as pd
from inspect_evals.hle.hle import get_verified_gold_uids
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1] / "plots"))
from plot_irt_pooled import OUT, responses  # noqa: E402

df = responses()
df = df[(df.benchmark != "hle") | df.uid.isin(get_verified_gold_uids())]
M = df.groupby(["benchmark", "item", "model"]).correct.mean().unstack("model")  # item x model
b = pd.read_csv(OUT / "pooled_gold_difficulties.csv").set_index(["benchmark", "item"]).b.loc[M.index]
rank = {"IRT": b.rank(), "P": (-M.mean(axis=1)).rank(method="first")}  # 1 = easiest
gpqa = M.index.get_level_values(0) == "gpqa"
n = len(M)


def split(r, scheme):
    if scheme == "easy GPQA / HLE":
        rg = r[gpqa].rank()
        return set(rg[rg <= round(len(rg) / 4)].index), set(M.index[~gpqa])
    if scheme == "all GPQA / HLE":
        return set(M.index[gpqa]), set(M.index[~gpqa])
    q = round(n / 4) if scheme.startswith("25") else round(n / 2)
    easy = set(r[r <= q].index)
    hard = set(r[r > n - q].index) if scheme == "25 / 25" else set(M.index) - easy
    return easy, hard


rows = []
for scheme in ["25 / 25", "25 / rest", "50 / 50", "easy GPQA / HLE", "all GPQA / HLE"]:
    sets = {k: split(r, scheme) for k, r in rank.items()}
    e_ov = len(sets["IRT"][0] & sets["P"][0]) / len(sets["IRT"][0])
    h_ov = len(sets["IRT"][1] & sets["P"][1]) / len(sets["IRT"][1])
    for k, (easy, hard) in sets.items():
        ae, ah = M.loc[list(easy)].mean(), M.loc[list(hard)].mean()
        ok = ae.notna() & ah.notna()
        rows.append(dict(split=scheme, builder=k, n_easy=len(easy), gpqa_in_easy=sum(i[0] == "gpqa" for i in easy),
                         n_hard=len(hard), easy_overlap=e_ov, hard_overlap=h_ov,
                         acc_easy=ae.mean(), acc_hard=ah.mean(), n_models=ok.sum(),
                         r=stats.pearsonr(ae[ok], ah[ok])[0], rho=stats.spearmanr(ae[ok], ah[ok])[0]))
out = pd.DataFrame(rows)
out.to_csv(OUT / "easy_hard_splits.csv", index=False)
print(out.round(3).to_string(index=False))
