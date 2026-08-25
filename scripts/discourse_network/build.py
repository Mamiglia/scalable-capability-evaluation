"""Clause-network pipeline: frozen store + segmenter output -> nodes, node embeddings,
edges, per-transcript features, and the transcript / pair design-matrix tables.

  uv run python scripts/discourse_network/build.py

Runs segment.py shards in the parser venv if their outputs are missing, then each step
below, skipping a step whose output file already exists (delete the file to redo it):
  nodes.parquet, node_embeddings.npy, edges.parquet, transcript_features.parquet,
  pair_features.parquet, node_sae/<group>.npz, edge_sae/<group>.npz, manifest.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "verdict_dissection"))
from discourse_network.edges import cap_parents, discourse_edges, long_range_edges, tool_edges  # noqa: E402
from discourse_network.features import EdgeSpace, TopKSAE, sae_block, topology_block  # noqa: E402
from discourse_network.nodes import build_nodes  # noqa: E402
from features_basic import TERMS, _count  # noqa: E402

KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
SAE_NODE_KINDS = {"assistant_edu", "equation", "code", "answer", "tool_call"}   # model-authored nodes only


def transcript_key(row):
    return hashlib.sha256("|".join(str(row[k]) for k in KEYS).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- steps

def make_nodes(store, clauses, cfg):
    """One row per node, with A's keys and transcript_key; A1 backtracking/verification markers on clauses."""
    by_key = {k: g for k, g in clauses.groupby(KEYS, dropna=False)}
    rows = []
    for r in store.to_dict("records"):
        parts = json.loads(r["transcript_parts_json"])
        spans = {}
        for c in by_key.get(tuple(r[k] for k in KEYS), pd.DataFrame()).itertuples():
            spans.setdefault((c.part_id, c.subpart_id), []).append((c.start, c.end))
        nodes = build_nodes(r["question"], r["transcript"], parts, spans, cfg["tool_types"],
                            cfg["python_write_markers"], cfg["error_markers"])
        key = transcript_key(r)
        for n in nodes:
            text = n["text"] if n["kind"] == "assistant_edu" else ""
            n["backtracking_marker"] = _count(TERMS["backtracking"], text) > 0
            n["verification_marker"] = _count(TERMS["verification"], text) > 0
            rows.append({k: r[k] for k in KEYS} | {"transcript_key": key} | n)
    return pd.DataFrame(rows)


def embed_nodes(texts, cfg):
    """Unit-norm embedding per text with the pinned encoder. Texts over max_tokens are split into
    token chunks and mean-pooled. Chunk vectors are cached by text hash in cfg["cache_dir"], so a
    killed run resumes and rebuilt nodes only embed new text."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model"], revision=cfg["revision"], device=cfg["device"])
    model.max_seq_length = cfg["max_tokens"]
    chunks, owner = [], []
    for i, text in enumerate(texts):
        tokens = model.tokenizer.encode(text or " ", add_special_tokens=False)
        for j in range(0, max(len(tokens), 1), cfg["max_tokens"]):
            chunks.append(model.tokenizer.decode(tokens[j:j + cfg["max_tokens"]]) or " ")
            owner.append(i)
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    known = {}                                                        # text hash -> vector, from earlier blocks
    for path in sorted(cache.glob("block-*.npz")):
        z = np.load(path)
        known.update(zip(z["hashes"].tolist(), z["vectors"]))
    digest = [hashlib.sha1(c.encode()).hexdigest() for c in chunks]
    todo = sorted({d for d in digest if d not in known})
    text_of = dict(zip(digest, chunks))
    todo.sort(key=lambda d: len(text_of[d]))                          # length-sorted batches run faster
    block = cfg.get("block_size", 20000)
    for b, start in enumerate(range(0, len(todo), block), start=len(list(cache.glob("block-*.npz")))):
        part = todo[start:start + block]
        vectors = model.encode([text_of[d] for d in part], batch_size=cfg["batch_size"], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        np.savez(cache / f"block-{b:04d}.npz", hashes=np.array(part), vectors=vectors.astype(np.float32))
        known.update(zip(part, vectors))
        print(f"embedded chunks {min(start + block, len(todo))}/{len(todo)}", flush=True)
    dim = len(next(iter(known.values())))
    out = np.zeros((len(texts), dim), dtype=np.float32)
    np.add.at(out, np.asarray(owner), np.stack([known[d] for d in digest]))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def make_edges(store, nodes, embeddings, trees, cfg):
    trees_by_key = {k: g for k, g in trees.groupby(KEYS, dropna=False)}
    groups = {k: g for k, g in nodes.groupby("transcript_key")}
    rows = []
    for r in store.to_dict("records"):
        key = transcript_key(r)
        group = groups[key]
        local = group.to_dict("records")
        parts = json.loads(r["transcript_parts_json"])
        texts = {(pid, sid): (sp["content"], sp["start"]) for pid, part in enumerate(parts)
                 for sid, sp in enumerate(part.get("subparts", []))
                 if part["role"] == "assistant" and sp["kind"] == "content"}
        by_part = {}
        for t in trees_by_key.get(tuple(r[k] for k in KEYS), pd.DataFrame()).itertuples():
            by_part.setdefault((t.part_id, t.subpart_id), []).append(json.loads(t.tree_json))
        edges = (discourse_edges(local, by_part, texts) + tool_edges(local)
                 + long_range_edges(local, embeddings[group.index.to_numpy()], cfg["k_similar"],
                                    cfg["similarity_threshold"]))
        rows += [{"transcript_key": key} | e for e in cap_parents(edges, cfg["max_parents"])]
    return pd.DataFrame(rows)


def make_features(store, nodes, edges, embeddings, split, cfg, out_dir):
    """Transcript table: A's keys + topology block + node/edge SAE blocks (NaN outside the SAE's group)."""
    keys = store[KEYS].reset_index(drop=True)
    keys["transcript_key"] = [transcript_key(r) for r in store.to_dict("records")]
    position = {k: i for i, k in enumerate(keys.transcript_key)}
    node_groups = {k: g.to_dict("records") for k, g in nodes.groupby("transcript_key")}
    edge_groups = {k: g.to_dict("records") for k, g in edges.groupby("transcript_key")}
    topo = pd.DataFrame([topology_block(node_groups[k], edge_groups.get(k, [])) for k in keys.transcript_key])
    sae_blocks = pd.DataFrame(index=keys.index)                       # SAE columns, NaN outside each group
    dev = set(map(tuple, split.loc[split.split != "confirmation", ["bench", "item"]].itertuples(index=False)))
    nodes["row"] = np.arange(len(nodes))                              # row into the embedding matrix
    node_dev = np.array([(b, i) in dev for b, i in zip(nodes.bench, nodes.item)])
    sae = cfg["sae"]
    for group, benches in sae["benchmark_groups"].items():
        in_group = keys.bench.isin(benches).to_numpy()
        n_mask = (nodes.bench.isin(benches) & nodes.kind.isin(SAE_NODE_KINDS)).to_numpy()
        if (n_mask & node_dev).sum() < 2 * sae["width"]:
            continue
        node_sae = TopKSAE.fit(embeddings[n_mask & node_dev], sae["width"], sae["top_k"], sae["iterations"],
                               sae["batch_size"], sae["seed"])
        z = node_sae.transform(embeddings[n_mask])
        owner = nodes.loc[n_mask, "transcript_key"].map(position).to_numpy()
        sae_blocks = sae_blocks.combine_first(
            _only(sae_block(z, owner, len(keys), f"node_sae_{group}", sae["min_support"], node_dev[n_mask]),
                  in_group))
        e = edges[edges.transcript_key.map(lambda k: in_group[position[k]])]
        first_row = nodes.groupby("transcript_key")["row"].min()
        offset = e.transcript_key.map(first_row).to_numpy()
        pairs = [{"source_node_id": s, "target_node_id": t} for s, t in
                 zip(e.source_node_id.to_numpy() + offset, e.target_node_id.to_numpy() + offset)]
        edge_dev = node_dev[[p["source_node_id"] for p in pairs]]
        if edge_dev.sum() < 2 * sae["width"]:
            continue
        space = EdgeSpace.fit(embeddings, [p for p, d in zip(pairs, edge_dev) if d], sae["edge_pca_components"], sae["seed"])
        v = space.transform(embeddings, pairs)
        edge_sae = TopKSAE.fit(v[edge_dev], sae["width"], sae["top_k"], sae["iterations"], sae["batch_size"], sae["seed"])
        sae_blocks = sae_blocks.combine_first(
            _only(sae_block(edge_sae.transform(v), e.transcript_key.map(position).to_numpy(), len(keys),
                            f"edge_sae_{group}", sae["min_support"], edge_dev), in_group))
        for name, model in (("node_sae", node_sae), ("edge_sae", edge_sae)):
            (out_dir / name).mkdir(exist_ok=True)
            np.savez(out_dir / name / f"{group}.npz", decoder=model.decoder, top_k=model.top_k)
    return pd.concat([keys, topo, sae_blocks[sorted(sae_blocks.columns)]], axis=1)


def _only(block, in_group):
    """NaN for transcripts outside the SAE's benchmark group."""
    block.loc[~in_group, :] = np.nan
    return block


# ---------------------------------------------------------------- pipeline

def run(store, clauses, trees, split, verdicts, cfg, out_dir, embed=embed_nodes, stop_after=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = store[store.correct & store.is_canonical].sort_values(KEYS).reset_index(drop=True)
    if not (out_dir / "nodes.parquet").exists():
        make_nodes(store, clauses, cfg).to_parquet(out_dir / "nodes.parquet", index=False)
    nodes = pd.read_parquet(out_dir / "nodes.parquet")
    if stop_after == "nodes":
        return nodes
    if not (out_dir / "node_embeddings.npy").exists():
        encoder_cfg = cfg["encoder"] | {"cache_dir": out_dir / "embed_cache"}
        np.save(out_dir / "node_embeddings.npy", embed(nodes.text.fillna("").tolist(), encoder_cfg))
    embeddings = np.load(out_dir / "node_embeddings.npy")
    if stop_after == "embeddings":
        return embeddings
    if not (out_dir / "edges.parquet").exists():
        make_edges(store, nodes, embeddings, trees, cfg["edges"]).to_parquet(out_dir / "edges.parquet", index=False)
    edges = pd.read_parquet(out_dir / "edges.parquet")
    if stop_after == "edges":
        return edges
    if not (out_dir / "transcript_features.parquet").exists():
        make_features(store, nodes, edges, embeddings, split, cfg, out_dir).to_parquet(
            out_dir / "transcript_features.parquet", index=False)
    features = pd.read_parquet(out_dir / "transcript_features.parquet")
    if stop_after == "features":
        return features
    from aggregate_surprisal import build_pairs
    pairs = build_pairs(verdicts, features.drop(columns=["transcript_key"]))
    pairs.to_parquet(out_dir / "pair_features.parquet", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "freeze": cfg.get("freeze"), "segmenter": cfg.get("segmenter"), "encoder": cfg.get("encoder"),
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "rows": {"transcripts": len(features), "nodes": len(nodes), "edges": len(edges), "pairs": len(pairs)},
        "columns": {"topology": sum(c.startswith("topo_") for c in features),
                    "node_sae": sum(c.startswith("node_sae_") for c in features),
                    "edge_sae": sum(c.startswith("edge_sae_") for c in features)}}, indent=2))
    return features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "workstream_c.yaml")
    ap.add_argument("--stop-after", choices=["nodes", "embeddings", "edges", "features"])
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir, freeze = ROOT / cfg["out_dir"], ROOT / cfg["freeze"]
    segments, shards = out_dir / "segments", cfg["segmenter"]["shards"]
    missing = [i for i in range(shards) if not (segments / f"shard-{i}.clauses.parquet").exists()]
    if missing:
        python = Path(cfg["segmenter"]["python"]).expanduser()
        procs = [subprocess.Popen([str(python), str(Path(__file__).with_name("segment.py")),
                                   "--shard", str(i), "--n-shards", str(shards)]) for i in missing]
        if any(p.wait() for p in procs):
            raise SystemExit("a segmentation shard failed")
    clauses = pd.concat([pd.read_parquet(p) for p in sorted(segments.glob("shard-*.clauses.parquet"))])
    trees = pd.concat([pd.read_parquet(p) for p in sorted(segments.glob("shard-*.trees.parquet"))])
    verdicts = pd.DataFrame() if args.stop_after else pd.read_parquet(ROOT / cfg["verdicts"])
    result = run(pd.read_parquet(freeze / "transcript_store.parquet"), clauses, trees,
                 pd.read_csv(freeze / "item_split.csv"), verdicts, cfg, out_dir, stop_after=args.stop_after)
    if args.stop_after:
        print(f"{args.stop_after}: {len(result)} rows -> {out_dir}")
    else:
        print(f"{len(result)} transcripts x {len(result.columns) - len(KEYS) - 1} features -> {out_dir}")


if __name__ == "__main__":
    main()
