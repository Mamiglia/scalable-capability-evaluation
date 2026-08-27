"""Prepare cached, dry-run-only Message Batch requests for retained SAE concepts.

No network client is imported here.  The output is JSONL suitable for a separately
approved submission step; this command only reconstructs activations and writes requests.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from discourse_network.build import KEYS, SAE_NODE_KINDS  # noqa: E402
from discourse_network.features import EdgeSpace, TopKSAE  # noqa: E402

DESCRIPTION_MODEL = "claude-sonnet-5"
FIDELITY_MODEL = "claude-haiku-4-5"
BATCH_PRICES = {DESCRIPTION_MODEL: (1.0, 5.0), FIDELITY_MODEL: (0.5, 2.5)}
HARD_CEILING_USD = 16.0


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def cache_path(cache_dir, prompt):
    return Path(cache_dir) / f"{prompt_sha256(prompt)}.json"


def _tokens(text):
    """Conservative local preflight count; no API/tokeniser/network dependency."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def build_requests(prompts, cache_dir, ceiling=HARD_CEILING_USD):
    """Return uncached Anthropic batch requests and a cost preflight."""
    requests, input_tokens, output_tokens, cached = [], {}, {}, 0
    for custom_id, prompt, model, max_tokens in prompts:
        if cache_path(cache_dir, prompt).exists():
            cached += 1
            continue
        requests.append({"custom_id": custom_id, "params": {"model": model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]}})
        input_tokens[model] = input_tokens.get(model, 0) + _tokens(prompt)
        output_tokens[model] = output_tokens.get(model, 0) + max_tokens
    cost = sum((input_tokens.get(m, 0) * price[0] + output_tokens.get(m, 0) * price[1]) / 1e6
               for m, price in BATCH_PRICES.items())
    estimate = {"requests": len(requests), "cached": cached, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "projected_usd": round(cost, 6), "ceiling_usd": ceiling}
    if cost > ceiling:
        raise RuntimeError(f"projected batch cost ${cost:.2f} exceeds ${ceiling:.2f} ceiling")
    return requests, estimate


def conservative_full_preflight(n_concepts, ceiling=HARD_CEILING_USD):
    """Approved upper assumptions: description 6k/100; fidelity 500/4, batch prices."""
    description = n_concepts * (6000 * 1.0 + 100 * 5.0) / 1e6
    fidelity = n_concepts * 200 * (500 * 0.5 + 4 * 2.5) / 1e6
    total = description + fidelity
    if total > ceiling:
        raise RuntimeError(f"projected batch cost ${total:.2f} exceeds ${ceiling:.2f} ceiling")
    return {"concepts": n_concepts, "description_requests": n_concepts,
            "fidelity_requests": n_concepts * 200, "projected_usd": round(total, 3),
            "ceiling_usd": ceiling}


def reconstruct_edge_space(nodes, edges, embeddings, split, benches, components, seed=0):
    """Exactly reconstruct build.py's scaler/PCA fit from development edges."""
    local_edges = edges[edges.transcript_key.isin(nodes.loc[nodes.bench.isin(benches), "transcript_key"])]
    first_row = nodes.groupby("transcript_key", sort=False).apply(lambda x: x.index.min(), include_groups=False)
    offsets = local_edges.transcript_key.map(first_row).to_numpy()
    pairs = [{"source_node_id": int(s + o), "target_node_id": int(t + o)} for s, t, o in
             zip(local_edges.source_node_id, local_edges.target_node_id, offsets)]
    dev_items = set(map(tuple, split.loc[split.split != "confirmation", ["bench", "item"]].itertuples(index=False)))
    owner = nodes.loc[[p["source_node_id"] for p in pairs]]
    development = np.array([(b, i) in dev_items for b, i in zip(owner.bench, owner.item)])
    space = EdgeSpace.fit(embeddings, [p for p, d in zip(pairs, development) if d], components, seed)
    return space, pairs, development


def select_top_examples(units, activations, n=10, kind="node"):
    """Highest activations, stable ties, with at most one unit per transcript."""
    order = np.lexsort((np.arange(len(units)), -np.asarray(activations)))
    rows, seen = [], set()
    by_transcript = {k: g.set_index("node_id") for k, g in units.groupby("transcript_key")} if kind == "node" else {}
    for i in order:
        row = units.iloc[int(i)]
        key = row.transcript_key
        if key in seen:
            continue
        if kind == "node":
            group, node_id = by_transcript[key], int(row.node_id)
            item = {"transcript_key": key, "text": str(row.text),
                    "previous": str(group.loc[node_id - 1].text) if node_id - 1 in group.index else None,
                    "next": str(group.loc[node_id + 1].text) if node_id + 1 in group.index else None}
        else:
            item = {"transcript_key": key, "source": str(row.source_text), "target": str(row.target_text)}
        rows.append(item)
        seen.add(key)
        if len(rows) == n:
            break
    if len(rows) != n:
        raise ValueError(f"need {n} distinct transcripts, found {len(rows)}")
    return rows


def choose_fidelity_examples(activations, concept_id, n_each=100):
    """Stable concept-specific sample of active and inactive unit row numbers."""
    z = np.asarray(activations)
    active, inactive = np.flatnonzero(z > 0), np.flatnonzero(z == 0)
    if len(active) < n_each or len(inactive) < n_each:
        raise ValueError(f"{concept_id}: need {n_each} active and inactive examples")
    seed = int.from_bytes(hashlib.sha256(concept_id.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    chosen = np.r_[rng.choice(active, n_each, replace=False), rng.choice(inactive, n_each, replace=False)]
    return chosen[rng.permutation(len(chosen))]


def description_result(description, answers, activations):
    """Accepted description JSON value, including activation/yes-no Pearson fidelity."""
    labels = np.array([str(answer).strip().lower() == "yes" for answer in answers], dtype=float)
    values = np.asarray(activations, dtype=float)
    if len(labels) != len(values) or not len(labels):
        raise ValueError("answers and activations must have the same non-zero length")
    fidelity = 0.0 if labels.std() == 0 or values.std() == 0 else float(np.corrcoef(labels, values)[0, 1])
    return {"description": str(description).strip(), "fidelity": fidelity, "n_examples": len(labels)}


def description_prompt(concept_id, examples):
    return ("Describe the property these text units share, as a concise concept. Do not mention activation "
            f"scores or feature IDs. Concept {concept_id}.\nExamples:\n" +
            "\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in examples))


def fidelity_prompt(description, unit):
    return (f"Does this unit have the property: {description}? Answer only yes or no.\nUnit: "
            + json.dumps(unit, ensure_ascii=False, sort_keys=True))


def _retained(features, prefix, group):
    stem = f"{prefix}_{group}_f"
    return sorted({int(c[len(stem):len(stem) + 2]) for c in features if c.startswith(stem) and c.endswith("_mean")})


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(out_dir, split, cfg, ceiling=HARD_CEILING_USD):
    """Reconstruct retained activations and write description requests; fidelity follows cached descriptions."""
    out_dir = Path(out_dir)
    nodes, edges = pd.read_parquet(out_dir / "nodes.parquet"), pd.read_parquet(out_dir / "edges.parquet")
    embeddings = np.load(out_dir / "node_embeddings.npy", mmap_mode="r")
    features = pd.read_parquet(out_dir / "transcript_features.parquet")
    cache = out_dir / "llm_cache"
    prompts, concepts = [], []
    sae_cfg = cfg["sae"]
    for group, benches in sae_cfg["benchmark_groups"].items():
        node_mask = nodes.bench.isin(benches) & nodes.kind.isin(SAE_NODE_KINDS)
        node_units = nodes.loc[node_mask, ["transcript_key", "node_id", "text"]].reset_index(drop=True)
        node_model = np.load(out_dir / "node_sae" / f"{group}.npz")
        node_z = TopKSAE(node_model["decoder"], int(node_model["top_k"])).transform(embeddings[node_mask])
        space, pairs, _ = reconstruct_edge_space(nodes, edges, embeddings, split, benches,
                                                  sae_cfg["edge_pca_components"], sae_cfg["seed"])
        edge_keys = edges[edges.transcript_key.isin(nodes.loc[nodes.bench.isin(benches), "transcript_key"])]
        edge_units = edge_keys[["transcript_key", "source_node_id", "target_node_id"]].reset_index(drop=True)
        text_lookup = nodes.set_index(["transcript_key", "node_id"]).text
        edge_units["source_text"] = [text_lookup.loc[(k, s)] for k, s in zip(edge_units.transcript_key, edge_units.source_node_id)]
        edge_units["target_text"] = [text_lookup.loc[(k, t)] for k, t in zip(edge_units.transcript_key, edge_units.target_node_id)]
        edge_model = np.load(out_dir / "edge_sae" / f"{group}.npz")
        edge_z = TopKSAE(edge_model["decoder"], int(edge_model["top_k"])).transform(space.transform(embeddings, pairs))
        for kind, units, z, prefix in (("node", node_units, node_z, "node_sae"),
                                       ("edge", edge_units, edge_z, "edge_sae")):
            for feature_id in _retained(features, prefix, group):
                concept_id = f"{kind}/{group}/f{feature_id:02d}"
                examples = select_top_examples(units, z[:, feature_id], 10, kind)
                prompt = description_prompt(concept_id, examples)
                prompts.append((f"description/{concept_id}", prompt, DESCRIPTION_MODEL, 100))
                concepts.append((concept_id, kind, units, z[:, feature_id], prompt))
    full = conservative_full_preflight(len(concepts), ceiling)
    description_requests, description_estimate = build_requests(prompts, cache, ceiling)
    fidelity_prompts = []
    for concept_id, kind, units, activation, prompt in concepts:
        cached = cache_path(cache, prompt)
        if not cached.exists():
            continue
        description = json.loads(cached.read_text())["text"].strip()
        for ordinal, i in enumerate(choose_fidelity_examples(activation, concept_id)):
            row = units.iloc[int(i)]
            unit = ({"text": str(row.text)} if kind == "node" else
                    {"source": str(row.source_text), "target": str(row.target_text)})
            fidelity_prompts.append((f"fidelity/{concept_id}/{ordinal:03d}", fidelity_prompt(description, unit),
                                     FIDELITY_MODEL, 4))
    fidelity_requests, fidelity_estimate = build_requests(fidelity_prompts, cache, ceiling)
    request_dir = out_dir / "batch_requests"
    _write_jsonl(request_dir / "descriptions.jsonl", description_requests)
    _write_jsonl(request_dir / "fidelity.jsonl", fidelity_requests)
    report = {"dry_run": True, "network_used": False, "full_preflight": full,
              "descriptions": description_estimate, "fidelity": fidelity_estimate,
              "note": "Fidelity requests are emitted only when the matching description is cached."}
    (request_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "workstream_c.yaml")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ceiling", type=float, default=HARD_CEILING_USD)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or ROOT / cfg["out_dir"]
    split = pd.read_csv(ROOT / cfg["freeze"] / "item_split.csv")
    print(json.dumps(prepare(out_dir, split, cfg, args.ceiling), indent=2))


if __name__ == "__main__":
    main()
