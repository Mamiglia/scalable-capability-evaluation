"""Extra dashboard data: t-SNE map + per-direction word clouds (local, cached).

Reuses the sentence embeddings cluster_features.py already cached (no model
load) and the sentences.json the maps exporter wrote. Adds two files the
sentences_ml viewer loads on top of its existing three:
  tsne.json      2-D t-SNE per non-trivial sentence + cluster + all <dir>_len.
  wordcloud.json per direction, distinctive up/down terms by Monroe log-odds.

Cluster ids/labels come from feature_clusters_ml.csv; the per-sentence cluster
is reproduced by re-running the same KMeans (identical cached embeddings, k,
seed) so ids line up with that CSV's labels.

Usage:
  uv run python scripts/analyses/judge_interp/export_dashboard_extras.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "interactive_viewer/data_ml"
CLUSTERS = ROOT / "judge_results/judge_interp/feature_clusters_ml.csv"
CACHE = Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge") \
    / "3b9bbeb6-5ca1-435b-a516-b3541bf624eb/scratchpad/sentence_emb.npy"

# LaTeX command fragments the alphabetic tokenizer leaves behind (\frac -> "frac").
# Not symbols, but not content either; boxed/option/final/step are deliberately kept.
LATEX = {"frac", "sqrt", "cdot", "times", "text", "mathrm", "mathbf", "left", "right",
         "begin", "end", "quad", "langle", "rangle", "hat", "vec", "tag", "label",
         "displaystyle", "textbf", "textit", "mathcal", "partial", "infty", "approx"}
STOP = list(ENGLISH_STOP_WORDS | LATEX)


def logodds_z(y_up, y_dn, prior):
    """Monroe et al. log-odds-ratio z-scores, up-group vs down-group.

    prior = informative Dirichlet pseudocounts (shape from the background
    corpus). z>0 = distinctive of up, z<0 = distinctive of down.
    """
    a0, n1, n2 = prior.sum(), y_up.sum(), y_dn.sum()
    d = (np.log(y_up + prior) - np.log(n1 + a0 - y_up - prior)) \
        - (np.log(y_dn + prior) - np.log(n2 + a0 - y_dn - prior))
    var = 1.0 / (y_up + prior) + 1.0 / (y_dn + prior)
    return d / np.sqrt(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--prior-mass", type=float, default=500.0,
                    help="total pseudocount of the informative Dirichlet prior")
    ap.add_argument("--top-terms", type=int, default=60, help="terms per cloud")
    args = ap.parse_args()

    recs = json.loads((DATA / "sentences.json").read_text())
    meta = json.loads((DATA / "meta.json").read_text())
    dirs = meta["directions"]
    df_full = pd.DataFrame(recs)
    emb_full = np.load(CACHE)
    assert len(emb_full) == len(df_full), "embedding cache is stale vs sentences.json"

    mask = (df_full["n_chars"] >= args.min_chars).to_numpy()
    df = df_full[mask].reset_index(drop=True)
    emb = emb_full[mask]
    print(f"{len(df)}/{len(df_full)} non-trivial sentences, embeddings {emb.shape}")

    # cluster ids/labels: same KMeans as cluster_features.py -> ids match the CSV
    cl = pd.read_csv(CLUSTERS)
    k = len(cl)
    labels = KMeans(k, random_state=args.seed, n_init=10).fit_predict(emb)
    label_of = {int(r.cluster_id): (r.label if isinstance(r.label, str) and r.label
                                    else f"cluster {int(r.cluster_id)}")
                for r in cl.itertuples()}

    # item-centre before t-SNE: sentences from the same GPQA item share topic
    # and vocabulary, so raw t-SNE clusters by item. Subtract each item's mean
    # embedding (across its model transcripts) to expose within-item structure.
    items = df["key"].str.rsplit("_", n=1).str[-1].to_numpy()
    emb_c = emb.astype(np.float64).copy()
    for it in np.unique(items):
        m = items == it
        emb_c[m] -= emb_c[m].mean(0)

    xy = TSNE(n_components=2, random_state=args.seed, init="pca",
              learning_rate="auto", perplexity=args.perplexity).fit_transform(emb_c)

    pts = []
    for i, r in df.iterrows():
        c = int(labels[i])
        p = dict(key=r["key"], sentence=r["sentence"],
                 x=round(float(xy[i, 0]), 3), y=round(float(xy[i, 1]), 3),
                 cluster_id=c, cluster_label=label_of.get(c, f"cluster {c}"))
        for d in dirs:
            p[f"{d}_len"] = round(float(r[f"{d}_len"]), 4)
        pts.append(p)
    (DATA / "tsne.json").write_text(json.dumps(pts))
    print(f"-> tsne.json ({len(pts)} points, {k} clusters)")

    # word clouds: distinctive terms of top-third vs bottom-third by <dir>_len
    vec = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(df["sentence"]).toarray()
    terms = np.array(vec.get_feature_names_out())
    bg = X.sum(0)
    prior = bg / bg.sum() * args.prior_mass

    clouds = {}
    for d in dirs:
        v = df[f"{d}_len"].to_numpy()
        lo, hi = np.quantile(v, [1 / 3, 2 / 3])
        up, dn = v >= hi, v <= lo
        z = logodds_z(X[up].sum(0), X[dn].sum(0), prior)
        order = np.argsort(z)
        pos = order[::-1][:args.top_terms]              # most up-distinctive
        neg = order[:args.top_terms]                    # most down-distinctive
        cnt_up, cnt_dn = X[up].sum(0), X[dn].sum(0)
        clouds[d] = dict(
            up=[dict(term=terms[j], z=round(float(z[j]), 3), count=int(cnt_up[j]))
                for j in pos if z[j] > 0],
            down=[dict(term=terms[j], z=round(float(z[j]), 3), count=int(cnt_dn[j]))
                  for j in neg if z[j] < 0])
    (DATA / "wordcloud.json").write_text(json.dumps(clouds))
    print(f"-> wordcloud.json ({', '.join(dirs)})")


if __name__ == "__main__":
    main()
