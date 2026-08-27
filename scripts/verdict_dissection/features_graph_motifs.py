"""Named graph-shape features of the clause network: fan-in, fan-out, chains, typed 3-node motifs,
directed 3-node motif census and one-step Weisfeiler-Lehman labels.

    uv run python scripts/verdict_dissection/features_graph_motifs.py

Reads the clause network's nodes and edges (all edge sources, duplicate pairs merged) and writes
<features_root>/raw/clause-motifs.parquet. Node types as in features_transitions (clause typed by
its discourse relation; equation, heading, answer, code, tool inspect/act, output ok/error).
Shape features are shares of nodes unless named *_max / *_count. Typed motifs are shares of the
motif's anchoring node (e.g. `after_tool_backtrack` = share of tool outputs followed by a
backtracking clause). WL labels are `wl__<type>|<sorted parent types>` shares, kept for labels
seen in >= 5% of transcripts.
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features_transitions import KEYS, NETWORK, node_types

ROOT = Path(__file__).resolve().parents[2]
EARLY = 0.2                                     # a "plan" node sits in the first fifth of the transcript
WL_MIN_PREVALENCE = 0.05


def graph_features(types, edges, markers):
    """One transcript: types[i] per node (node_id order), edges as (source, target) pairs, markers[i] in
    {'backtracking', 'verification', None}."""
    n = len(types)
    out, inc = defaultdict(list), defaultdict(list)
    for s, t in edges:
        out[s].append(t)
        inc[t].append(s)
    indeg = np.array([len(inc[i]) for i in range(n)])
    outdeg = np.array([len(out[i]) for i in range(n)])
    depth = [0] * n
    for i in range(n):
        depth[i] = 1 + max((depth[s] for s in inc[i]), default=-1)
    width = max(Counter(depth).values()) if n else 0
    answer = [i for i in range(n) if types[i] == "answer"]
    answer_in = max((indeg[a] for a in answer), default=0)
    # options converging: most same-type parents feeding one node
    sibling_convergence = max((max(Counter(types[s] for s in inc[i]).values(), default=0) for i in range(n)), default=0)
    # plan then execute: an early node with the largest out-degree, and how much of the graph descends from it
    early = [i for i in range(n) if i <= EARLY * n]
    plan = max(early, key=lambda i: outdeg[i], default=None)
    covered = set()
    if plan is not None and outdeg[plan] >= 2:
        stack = [plan]
        while stack:
            for t in out[stack.pop()]:
                if t not in covered:
                    covered.add(t)
                    stack.append(t)
    f = {"n_nodes": n, "n_edges": sum(outdeg),
         "in_degree_max": int(indeg.max(initial=0)), "out_degree_max": int(outdeg.max(initial=0)),
         "fan_in_share": float((indeg >= 2).mean()) if n else 0.0,
         "fan_out_share": float((outdeg >= 2).mean()) if n else 0.0,
         "sequential_share": float(((indeg <= 1) & (outdeg <= 1)).mean()) if n else 0.0,
         "depth_max": max(depth, default=0), "width_max": width,
         "depth_over_width": (max(depth, default=0) + 1) / max(width, 1),
         "answer_in_degree": int(answer_in),
         "sibling_convergence_max": int(sibling_convergence),
         "plan_fan_out": int(outdeg[plan]) if plan is not None else 0,
         "plan_coverage": len(covered) / max(n, 1)}
    # directed 3-node motif census, per node
    chain = fan_in = fan_out = feed_forward = 0
    for b in range(n):
        for a in inc[b]:
            for c in out[b]:
                if c in out[a]:
                    feed_forward += 1
                else:
                    chain += 1
        fan_in += len(inc[b]) * (len(inc[b]) - 1) // 2
        fan_out += len(out[b]) * (len(out[b]) - 1) // 2
    f |= {"motif_chain": chain / max(n, 1), "motif_fan_in": fan_in / max(n, 1),
          "motif_fan_out": fan_out / max(n, 1), "motif_feed_forward": feed_forward / max(n, 1)}
    # typed motifs: what follows a tool output, what surrounds an equation
    def share(anchor, hit):
        idx = [i for i in range(n) if anchor(i)]
        return sum(hit(i) for i in idx) / len(idx) if idx else 0.0
    is_output = lambda i: types[i].startswith("output_")
    children = lambda i: [t for t in out[i]]
    f |= {"after_tool_backtrack": share(is_output, lambda i: any(markers[t] == "backtracking" for t in children(i))),
          "after_tool_verify": share(is_output, lambda i: any(markers[t] == "verification" for t in children(i))),
          "after_tool_adversative": share(is_output, lambda i: any(types[t] == "adversative" for t in children(i))),
          "after_tool_another_tool": share(is_output, lambda i: any(types[t].startswith("tool_") for t in children(i))),
          "after_error_backtrack": share(lambda i: types[i] == "output_error",
                                         lambda i: any(markers[t] == "backtracking" for t in children(i))),
          "equation_explained": share(lambda i: types[i] == "equation",
                                      lambda i: any(not types[t].startswith(("equation", "tool_", "output_")) for t in children(i))),
          "equation_to_equation": share(lambda i: types[i] == "equation", lambda i: any(types[t] == "equation" for t in children(i))),
          "equation_from_clause": share(lambda i: types[i] == "equation", lambda i: any(types[s] not in ("equation",) for s in inc[i])),
          "heading_fan_out": share(lambda i: types[i] == "heading", lambda i: outdeg[i] >= 2),
          "backtrack_share": sum(m == "backtracking" for m in markers) / max(n, 1),
          "verify_share": sum(m == "verification" for m in markers) / max(n, 1)}
    wl = Counter(f"wl__{types[i]}|{','.join(sorted(types[s] for s in inc[i]))}" for i in range(n))
    return f, {k: v / max(n, 1) for k, v in wl.items()}


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    nodes = pd.read_parquet(NETWORK / "nodes.parquet", columns=KEYS + ["transcript_key", "node_id", "kind", "tool_type", "outcome",
                                                                          "backtracking_marker", "verification_marker"])
    edges = pd.read_parquet(NETWORK / "edges.parquet", columns=["transcript_key", "source_node_id", "target_node_id", "edge_source", "relation"])
    nodes = nodes.sort_values(["transcript_key", "node_id"]).reset_index(drop=True)
    nodes["type"] = node_types(nodes, edges)
    nodes["marker"] = np.where(nodes.backtracking_marker.fillna(False), "backtracking",
                               np.where(nodes.verification_marker.fillna(False), "verification", None))
    pairs = edges.drop_duplicates(["transcript_key", "source_node_id", "target_node_id"]).groupby("transcript_key")
    pair_lists = {k: list(zip(g.source_node_id, g.target_node_id)) for k, g in pairs}
    rows, wl_rows = [], []
    for key, group in nodes.groupby("transcript_key", sort=False):
        f, wl = graph_features(group.type.tolist(), pair_lists.get(key, []), group.marker.tolist())
        rows.append(dict(zip(KEYS, group.iloc[0][KEYS])) | f)
        wl_rows.append(wl)
    table = pd.DataFrame(rows)
    wl = pd.DataFrame(wl_rows).fillna(0.0)
    wl = wl[wl.columns[(wl > 0).mean() >= WL_MIN_PREVALENCE]]
    table = pd.concat([table, wl], axis=1)
    table.to_parquet(features_root / "raw/clause-motifs.parquet", index=False)
    named = [c for c in table.columns if c not in KEYS and not c.startswith("wl__")]
    print(f"{len(table)} transcripts, {len(named)} named features, {wl.shape[1]} WL labels")
    print(table.groupby("bench")[named].mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
