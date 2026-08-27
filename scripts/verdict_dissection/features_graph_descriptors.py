"""Graph-theoretic descriptors of each clause network: density, clustering, small-worldness, path
lengths, degree heterogeneity, centralisation, communities, edge-length distribution, entropies.

    uv run python scripts/verdict_dissection/features_graph_descriptors.py

All edge sources, duplicate pairs merged, the question node excluded. Undirected quantities use the
undirected version of the graph; path lengths are over its largest connected component. Small-
worldness is sigma = (C / C_rand) / (L / L_rand) with the Erdos-Renyi references C_rand = k / n and
L_rand = ln n / ln k for mean degree k. Writes <features_root>/raw/clause-descriptors.parquet.
"""
import math
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from features_transitions import KEYS, NETWORK, node_types

ROOT = Path(__file__).resolve().parents[2]
LONG_EDGE = 5


def entropy(counts):
    total = sum(counts)
    return -sum(c / total * math.log(c / total) for c in counts if c) if total else 0.0


def descriptors(types, edges, relations):
    """One transcript: node types in node_id order (question already dropped, ids 0..n-1),
    edges as (source, target) pairs on those ids, relations as edge labels."""
    n = len(types)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    u = g.to_undirected()
    m = u.number_of_edges()
    degrees = np.array([d for _, d in u.degree()]) if n else np.array([0])
    k = degrees.mean() if n else 0.0
    components = list(nx.connected_components(u)) if n else []
    giant = u.subgraph(max(components, key=len)) if components else u
    if giant.number_of_nodes() > 1:
        path_length = nx.average_shortest_path_length(giant)
        diameter = nx.diameter(giant)
    else:
        path_length, diameter = 0.0, 0
    clustering = nx.average_clustering(u) if n else 0.0
    c_rand = k / n if n and k else 0.0
    l_rand = math.log(n) / math.log(k) if n > 1 and k > 1 else 0.0
    sigma = (clustering / c_rand) / (path_length / l_rand) if c_rand and l_rand and path_length else 0.0
    lengths = np.array([t - s for s, t in edges]) if edges else np.array([0])
    in_deg = np.array([d for _, d in g.in_degree()]) if n else np.array([0])
    out_deg = np.array([d for _, d in g.out_degree()]) if n else np.array([0])
    if n > 2:
        bet = np.array(list(nx.betweenness_centrality(u, k=min(n, 60), seed=0).values()))
        centralisation = (bet.max() - bet).sum() / (n - 1)
        try:
            assortativity = nx.degree_assortativity_coefficient(u)
        except (ValueError, ZeroDivisionError):
            assortativity = 0.0
        communities = list(nx.community.greedy_modularity_communities(u)) if m else [set(range(n))]
        modularity = nx.community.modularity(u, communities) if m else 0.0
    else:
        centralisation, assortativity, communities, modularity = 0.0, 0.0, [], 0.0
    return {"density": 2 * m / (n * (n - 1)) if n > 1 else 0.0,
            "cyclomatic": m - n + len(components),
            "mean_degree": float(k), "degree_cv": float(degrees.std() / k) if k else 0.0,
            "max_degree_share": float(degrees.max() / max(m, 1)) if n else 0.0,
            "assortativity": 0.0 if np.isnan(assortativity) else float(assortativity),
            "clustering": float(clustering), "transitivity": float(nx.transitivity(u)) if n > 2 else 0.0,
            "small_world_sigma": float(sigma),
            "path_length_mean": float(path_length), "diameter": int(diameter),
            "giant_component_share": len(giant) / n if n else 0.0, "components_per_node": len(components) / n if n else 0.0,
            "betweenness_centralisation": float(centralisation),
            "communities_per_node": len(communities) / n if n else 0.0, "modularity": float(modularity),
            "source_share": float((in_deg == 0).mean()), "sink_share": float((out_deg == 0).mean()),
            "edge_length_mean": float(lengths.mean()), "edge_length_median": float(np.median(lengths)),
            "edge_length_max": int(lengths.max()), "edge_length_cv": float(lengths.std() / lengths.mean()) if lengths.mean() else 0.0,
            "adjacent_edge_share": float((lengths == 1).mean()) if edges else 0.0,
            "long_edge_share": float((lengths > LONG_EDGE).mean()) if edges else 0.0,
            "edge_length_entropy": entropy(Counter(np.minimum(lengths, 20).tolist()).values()),
            "type_entropy": entropy(Counter(types).values()),
            "relation_entropy": entropy(Counter(relations).values())}


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    nodes = pd.read_parquet(NETWORK / "nodes.parquet", columns=KEYS + ["transcript_key", "node_id", "kind", "tool_type", "outcome"])
    edges = pd.read_parquet(NETWORK / "edges.parquet", columns=["transcript_key", "source_node_id", "target_node_id", "edge_source", "relation"])
    nodes = nodes.sort_values(["transcript_key", "node_id"]).reset_index(drop=True)
    nodes["type"] = node_types(nodes, edges)
    edges = edges[(edges.source_node_id > 0)]                      # drop the question node's edges; ids shift by one below
    edges = edges.drop_duplicates(["transcript_key", "source_node_id", "target_node_id"])
    edges["relation"] = edges.relation.fillna(edges.edge_source)
    by_key = {k: g for k, g in edges.groupby("transcript_key")}
    rows = []
    for key, group in nodes.groupby("transcript_key", sort=False):
        types = group.type.tolist()[1:]                               # node 0 is the question
        e = by_key.get(key)
        pairs = list(zip(e.source_node_id - 1, e.target_node_id - 1)) if e is not None else []
        rows.append(dict(zip(KEYS, group.iloc[0][KEYS])) | descriptors(types, pairs, e.relation.tolist() if e is not None else []))
    table = pd.DataFrame(rows)
    table.to_parquet(features_root / "raw/clause-descriptors.parquet", index=False)
    print(f"{len(table)} transcripts, {table.shape[1] - len(KEYS)} descriptors")
    print(table.groupby("bench").mean(numeric_only=True).round(2).T.to_string())


if __name__ == "__main__":
    main()
