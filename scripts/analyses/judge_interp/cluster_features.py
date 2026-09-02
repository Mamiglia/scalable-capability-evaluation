"""Data-discovered transcript features per judge direction.

No hand-picked feature list. Embed every sentence with the viewer's encoder
(Qwen3-Embedding-0.6B), k-means the embeddings, and per cluster report the
mean length-adjusted occlusion delta on each direction. Clusters = features
that emerged from the data. Rank by v_pref_gen_orth (taste beyond capability)
and v_cap (capability); positive delta = the sentence drives that direction UP.

Writes judge_results/judge_interp/feature_clusters_ml.csv.

Usage:
  uv run python scripts/analyses/judge_interp/cluster_features.py \
      [--k-grid 20 25 30 35 40] [--labels labels.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "interactive_viewer/data_ml/sentences.json"
META = ROOT / "interactive_viewer/data_ml/meta.json"
OUT = ROOT / "judge_results/judge_interp/feature_clusters_ml.csv"
CACHE = Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge") \
    / "3b9bbeb6-5ca1-435b-a516-b3541bf624eb/scratchpad/sentence_emb.npy"


def embed(sentences, encoder="Qwen/Qwen3-Embedding-0.6B", batch=16):
    """Reuse the viewer's encoder; cache to disk so re-runs skip the GPU."""
    if CACHE.exists():
        emb = np.load(CACHE)
        if len(emb) == len(sentences):
            return emb
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(encoder)
    emb = enc.encode(sentences, batch_size=batch, show_progress_bar=True,
                     normalize_embeddings=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, emb)
    return emb


def pick_k(emb, grid, seed=0):
    """Best silhouette within the requested range."""
    scores = {}
    for k in grid:
        lbl = KMeans(k, random_state=seed, n_init=10).fit_predict(emb)
        scores[k] = silhouette_score(emb, lbl)
    for k, s in scores.items():
        print(f"  k={k:>2}  silhouette={s:.4f}")
    return max(scores, key=scores.get)


def eta_sq(values, labels):
    """Share of a direction's driver-variance the clustering explains."""
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_between = sum(len(g := values[labels == c]) * (g.mean() - grand) ** 2
                     for c in np.unique(labels))
    return ss_between / ss_tot if ss_tot else 0.0


def top_terms(sentences, labels, cluster, vec, X, n=6):
    """Terms most over-represented in a cluster vs the rest (data-driven hint)."""
    mask = labels == cluster
    diff = np.asarray(X[mask].mean(0)).ravel() - np.asarray(X[~mask].mean(0)).ravel()
    idx = diff.argsort()[::-1][:n]
    terms = np.array(vec.get_feature_names_out())[idx]
    return " ".join(terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-grid", type=int, nargs="+", default=[20, 25, 30, 35, 40])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=25,
                    help="drop fragments shorter than this; sentences.json is NOT pre-filtered")
    ap.add_argument("--labels", type=Path, help="json {cluster_id: label} to fill the CSV")
    args = ap.parse_args()

    recs = json.loads(DATA.read_text())
    meta = json.loads(META.read_text())
    dirs = meta["directions"]
    df_full = pd.DataFrame(recs)
    emb_full = embed(df_full["sentence"].tolist())   # cache covers every sentence

    # sentences.json ships unfiltered (bare "1.", "Given:", ...); drop the fragments.
    mask = (df_full["n_chars"] >= args.min_chars).to_numpy()
    df = df_full[mask].reset_index(drop=True)
    emb = emb_full[mask]
    print(f"{len(df)}/{len(df_full)} sentences kept (n_chars>={args.min_chars}), "
          f"{df['key'].nunique()} transcripts, embeddings {emb.shape}")

    print("k selection (silhouette):")
    k = pick_k(emb, args.k_grid, args.seed)
    print(f"chosen k = {k}")
    labels = KMeans(k, random_state=args.seed, n_init=10).fit_predict(emb)
    df["cluster"] = labels

    # stability: relabel with a second seed, Adjusted Rand Index
    lab2 = KMeans(k, random_state=args.seed + 1, n_init=10).fit_predict(emb)
    ari = adjusted_rand_score(labels, lab2)
    print(f"cluster stability (ARI across seeds) = {ari:.3f}")

    # variance of each direction's driver explained by cluster membership
    print("eta^2 (driver-variance explained by clusters):")
    for d in dirs:
        e = eta_sq(df[f"{d}_len"].to_numpy(), labels)
        assert 0.0 <= e <= 1.0, f"eta^2 out of range: {d}={e}"  # guards the SS ratio
        print(f"  {d:<20} {e:.3f}")

    # data-driven term hints per cluster
    vec = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2))
    X = vec.fit_transform(df["sentence"])

    # centroids for representative-exemplar selection (emb is normalized)
    labmap = {c: emb[labels == c].mean(0) for c in range(k)}

    override = json.loads(args.labels.read_text()) if args.labels else {}
    rows = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        cen = labmap[c]
        sims = sub[["sentence"]].assign(
            sim=emb[labels == c] @ cen).sort_values("sim", ascending=False)
        exemplars = [s[:180] for s in sims["sentence"].head(4)]
        row = dict(cluster_id=c, size=len(sub),
                   auto_terms=top_terms(df["sentence"], labels, c, vec, X),
                   label=override.get(str(c), ""))
        for d in dirs:
            v = sub[f"{d}_len"].to_numpy()
            row[f"{d}_mean"] = round(v.mean(), 3)
            row[f"{d}_se"] = round(v.std(ddof=1) / np.sqrt(len(v)), 3) if len(v) > 1 else np.nan
        row["exemplars"] = " ||| ".join(exemplars)
        rows.append(row)
    res = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")

    # readable ranking for the two headline directions
    for d in ("v_pref_gen_orth", "v_cap"):
        print(f"\n===== ranked by {d}_mean (data-discovered features) =====")
        r = res.sort_values(f"{d}_mean", ascending=False)
        for _, x in r.iterrows():
            flag = "*" if abs(x[f"{d}_mean"]) > 2 * (x[f"{d}_se"] or 9e9) else " "
            print(f"{flag}c{int(x.cluster_id):>2} n={int(x['size']):>4} "
                  f"{d}={x[f'{d}_mean']:+.2f}±{x[f'{d}_se']:.2f}  "
                  f"vcap={x['v_cap_mean']:+.2f}  terms[{x.auto_terms}]")
        # exemplars for the top/bottom 3
        print("  --- top drivers UP ---")
        for _, x in r.head(3).iterrows():
            print(f"  c{int(x.cluster_id)} [{x.auto_terms}]")
            for e in x.exemplars.split(" ||| "):
                print(f"      {e}")
        print("  --- top drivers DOWN ---")
        for _, x in r.tail(3).iterrows():
            print(f"  c{int(x.cluster_id)} [{x.auto_terms}]")
            for e in x.exemplars.split(" ||| "):
                print(f"      {e}")


if __name__ == "__main__":
    main()
