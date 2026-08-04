"""Embed transcripts.

texts.parquet -> embeddings.parquet, one vector per unique text. The
encoder comes from the config: provider "local" (sentence-transformers,
free) or "voyage" (API, paid). Dry run by default -- prints volume and
cost, writes nothing; pass --run to embed.

Usage:
  uv run python scripts/analyses/score_prediction/embed.py        # dry run
  uv run python scripts/analyses/score_prediction/embed.py --run
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CHARS_PER_TOKEN = 3.7  # rough average, dry-run estimate only


def load_done(out_path: Path) -> pd.DataFrame | None:
    """Load existing embeddings"""
    if not out_path.exists():
        return None
    return pd.read_parquet(out_path)


def _local_encoder(encoder: dict):
    import torch
    from sentence_transformers import SentenceTransformer
    # Weights dtype from config; the sentence-transformers default is fp32,
    # which is ~30 GiB for the 8B encoder.
    model_kwargs = ({"torch_dtype": getattr(torch, encoder["dtype"])}
                    if encoder.get("dtype") else {})
    model = SentenceTransformer(encoder["name"], device=encoder.get("device"),
                                model_kwargs=model_kwargs)
    dim = model.get_embedding_dimension()
    assert dim == encoder["output_dim"], f"model dim {dim} != config {encoder['output_dim']}"

    def embed_batch(batch_texts):
        vectors = model.encode(batch_texts, batch_size=len(batch_texts),
                               normalize_embeddings=True)
        if model.device.type == "mps":
            torch.mps.empty_cache()  # long texts fragment Metal memory
        n_tokens = [len(model.tokenizer(t)["input_ids"]) for t in batch_texts]
        return vectors, n_tokens
    return embed_batch


def _voyage_encoder(encoder: dict):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import voyageai
    client = voyageai.Client()  # needs VOYAGE_API_KEY

    def embed_batch(batch_texts):
        result = client.embed(batch_texts, model=encoder["name"],
                              output_dimension=encoder["output_dim"])
        return result.embeddings, [np.nan] * len(batch_texts)
    return embed_batch


def embed_all(texts: pd.DataFrame, encoder: dict, out_path: Path,
              limit: int | None = None) -> pd.DataFrame | None:
    """Embed texts not already in out_path; checkpoint after every batch."""
    done = load_done(out_path)
    done_ids = set(done["text_id"]) if done is not None else set()
    # Sorting by length keeps batches padding-light on the local backend.
    todo = texts[~texts["text_id"].isin(done_ids)].sort_values("n_chars")
    if limit:
        todo = todo.head(limit)
    print(f"{len(done_ids)} already embedded, {len(todo)} to go")
    if not len(todo):
        return done

    embed_batch = (_local_encoder if encoder["provider"] == "local"
                   else _voyage_encoder)(encoder)
    frames = [] if done is None else [done]
    batch_size = encoder["batch_size"]
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        vectors, n_tokens = embed_batch(list(batch["text"]))
        frames.append(pd.DataFrame(dict(
            text_id=batch["text_id"].values,
            embedding=[np.asarray(v, dtype=np.float32) for v in vectors],
            n_tokens=n_tokens, encoder=encoder["name"])))
        pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
        print(f"\r{min(start + batch_size, len(todo))}/{len(todo)} embedded",
              end="", flush=True)
    print()
    return pd.concat(frames, ignore_index=True)


def run_checks(texts: pd.DataFrame, embeddings: pd.DataFrame, encoder: dict,
               seed: int, n_redo: int = 5, n_queries: int = 3, n_neighbours: int = 3):
    """Post-embedding sanity checks: coverage, norms, determinism, neighbours."""
    assert set(embeddings["text_id"]) == set(texts["text_id"]), "text_id mismatch"
    merged = embeddings.merge(texts[["text_id", "model", "item", "text"]], on="text_id")
    vectors = np.stack(merged["embedding"].to_numpy())
    norms = np.linalg.norm(vectors, axis=1)
    print(f"{len(merged)} embeddings, dim {vectors.shape[1]}, "
          f"norms {norms.min():.4f}-{norms.max():.4f} (expect ~1)")

    sample = merged.sample(n_redo, random_state=seed)
    embed_batch = (_local_encoder if encoder["provider"] == "local"
                   else _voyage_encoder)(encoder)
    fresh = np.stack([np.asarray(v) for v in embed_batch(list(sample["text"]))[0]])
    stored = np.stack(sample["embedding"].to_numpy())
    cosines = (fresh * stored).sum(1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1))
    print(f"determinism: {n_redo} re-embeds, min cosine {cosines.min():.6f} (expect >= 0.999)")

    print(f"nearest neighbours ({n_queries} random texts; expect same model or same item):")
    # Renormalize: MPS output norms drift up to ~0.4%, enough to break
    # cosine ordering and the skip-self assumption.
    unit = vectors / norms[:, None]
    sims = unit @ unit.T
    rng = np.random.default_rng(seed)
    for idx in rng.choice(len(merged), n_queries, replace=False):
        row = merged.iloc[idx]
        print(f"  {row['model']}  {row['item']}")
        for neighbour_idx in [i for i in np.argsort(sims[idx])[::-1] if i != idx][:n_neighbours]:
            neighbour = merged.iloc[neighbour_idx]
            print(f"    {sims[idx, neighbour_idx]:.3f}  {neighbour['model']}  "
                  f"{neighbour['item']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--run", action="store_true", help="embed for real (default: dry run)")
    ap.add_argument("--limit", type=int, help="embed only the first N missing texts")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    encoder = cfg["encoder"]
    out = ROOT / cfg["out_dir"]

    texts = pd.read_parquet(out / "texts.parquet")
    est_tokens = texts["n_chars"].sum() / CHARS_PER_TOKEN
    print(f"{len(texts)} texts, {texts['n_chars'].sum():,} chars, "
          f"~{est_tokens / 1e6:.2f}M tokens (chars/{CHARS_PER_TOKEN})")
    print(f"encoder: {encoder['provider']}:{encoder['name']}, dim {encoder['output_dim']}")
    if encoder["provider"] == "voyage":
        print(f"estimated cost ~${est_tokens / 1e6 * encoder['price_per_m_tokens']:.2f}")
    else:
        print("local encoder: free")
    if not args.run:
        print("dry run -- pass --run to embed")
        return

    out_path = out / "embeddings.parquet"
    embeddings = embed_all(texts, encoder, out_path, limit=args.limit)
    n_done = 0 if embeddings is None else len(embeddings)
    if embeddings is not None and n_done == texts["text_id"].nunique():
        run_checks(texts, embeddings, encoder, seed=cfg["seeds"][0])
    else:
        print(f"partial: {n_done}/{texts['text_id'].nunique()} embedded -- "
              "checks run when complete")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()