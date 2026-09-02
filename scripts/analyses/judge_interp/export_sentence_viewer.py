"""Export the sentence attribution data for the interactive viewer.

One record per sentence: its contribution to each axis (length-corrected,
and also centred within its transcript for fair ranking), the author
model, and its position in the transcript so the viewer can rebuild the
transcript with sentences shaded by loading.

Also lays the sentences out in 2D with sentence embeddings, then
t-SNE, so nearby points share meaning. The encoder is the one the
project already uses, cached locally.


Writes interactive_viewer/data/sentences/{sentences.json, meta.json}.

Usage:
  uv run python scripts/analyses/judge_interp/export_sentence_viewer.py \
      --maps <token_maps dir> [--min-chars 25]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[3]


def residualise(df, axis, within):
    x, y = df["n_chars"].to_numpy(float), df[axis].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    if not within:
        return resid
    off = pd.Series(resid).groupby(
        [df["text_id"].to_numpy(), df["item"].to_numpy()]).transform("mean")
    return resid - off.to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B",
                    help="sentence encoder for the semantic map; cached locally")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    elo = pd.read_csv(out / "elo.csv")
    model_of = {(t, i): m for t, i, m in zip(elo["text_id"], elo["item"], elo["model"])}

    rows, transcripts = [], {}
    for path in sorted(args.maps.glob("*.json")):
        d = json.loads(path.read_text())
        key = f"{d['text_id']}_{d['item']}"
        transcripts[key] = dict(text_id=d["text_id"], item=d["item"],
                                model=model_of.get((d["text_id"], d["item"]), "?"),
                                elo=round(float(d.get("elo", 0.0)), 3),
                                n_sentences=len(d["occlusion"]))
        for pos, occ in enumerate(d["occlusion"]):
            rec = dict(key=key, pos=pos, sentence=occ["sentence"],
                       n_chars=occ.get("n_chars", len(occ["sentence"])),
                       model=transcripts[key]["model"], elo=transcripts[key]["elo"])
            for axis, delta in occ["delta_last"].items():
                rec[axis] = -delta
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no maps under {args.maps}")
    df["text_id"] = df["key"].str.split("_").str[0]
    df["item"] = df["key"].str.split("_").str[1]

    axes = [c for c in ("v_pref_judge", "v_pref", "v_pref_orth", "v_cap", "approval")
            if c in df.columns]
    full = df.copy()
    for axis in axes:                        # keep every sentence for the transcript view
        full[f"{axis}_res"] = residualise(full, axis, within=True)
        full[f"{axis}_len"] = residualise(full, axis, within=False)
    plot = full[full["n_chars"] >= args.min_chars].copy()   # fragments are list markers

    # Semantic layout: sentence embeddings -> t-SNE. Nearby = similar meaning.
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.encoder)
    emb = enc.encode(plot["sentence"].tolist(), batch_size=args.batch_size,
                     show_progress_bar=True, normalize_embeddings=True)
    xy = TSNE(n_components=2, random_state=args.seed, init="pca",
              perplexity=min(30, max(5, len(emb) // 100))).fit_transform(emb)
    plot["tx"], plot["ty"] = xy[:, 0].round(2), xy[:, 1].round(2)

    keep = ["key", "pos", "sentence", "n_chars", "model", "elo", "tx", "ty"] + \
           [f"{a}_res" for a in axes] + [f"{a}_len" for a in axes]
    plot_out = plot[keep].round(3)
    dest = ROOT / "interactive_viewer/data/sentences"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sentences.json").write_text(plot_out.to_json(orient="records"))
    # transcript view needs every sentence, including the short fragments
    tcols = ["key", "pos", "sentence", "n_chars"] + [f"{a}_res" for a in axes]
    (dest / "transcripts.json").write_text(json.dumps(dict(
        meta=transcripts,
        sentences=json.loads(full[tcols].round(3).to_json(orient="records")))))
    (dest / "meta.json").write_text(json.dumps(dict(
        axes=axes, n_plot=len(plot_out), n_all=len(full),
        n_transcripts=len(transcripts),
        models=sorted(plot["model"].unique().tolist()),
        min_chars=args.min_chars)))
    print(f"{len(plot_out)} plotted sentences ({len(full)} incl. fragments), "
          f"{len(transcripts)} transcripts, {plot['model'].nunique()} models")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
