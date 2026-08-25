"""Automatic edges of a transcript graph. Every edge goes from an earlier node to a later
one. Three sources:
  discourse   RST tree within one assistant text part, one edge per internal tree node
              between the heads of its two children (head = nucleus child's head)
  tool        tool_call -> its tool_output (same call_id); tool_output -> first assistant
              node of the next assistant part
  similarity / identifier
              earlier nodes from other parts: top-k by embedding cosine above a threshold,
              and the nearest one sharing an identifier token
Edge fields: source_node_id, target_node_id, edge_source, relation, nuclearity, cosine.
"""
import re

import numpy as np

from .nodes import TARGETS, mask_prose

IDENTIFIER = re.compile(r"\b(?:[A-Za-z_]\w*\d\w*|[A-Za-z_]\w*_[A-Za-z0-9_]*)\b")   # foo_bar, x1, test_fileinput
STRUCTURAL = {"discourse", "tool"}
RANK = {"discourse": 0, "tool": 1, "identifier": 2, "similarity": 3}              # who wins a duplicate pair


def _edge(source, target, edge_source, relation=None, nuclearity=None, cosine=None):
    assert source < target, (source, target)
    return {"source_node_id": source, "target_node_id": target, "edge_source": edge_source,
            "relation": relation, "nuclearity": nuclearity, "cosine": cosine}


def discourse_edges(nodes, trees_by_part, texts_by_part):
    """trees_by_part: {(part_id, subpart_id): [tree dict, ...]} in masked coordinates of that
    subpart's text; texts_by_part: {(part_id, subpart_id): (text, transcript offset)}."""
    edges = []
    for key, trees in trees_by_part.items():
        text, base = texts_by_part[key]
        _, cmap, _ = mask_prose(text)
        local = [n for n in nodes if n["part_id"] == key[0] and n["start"] is not None
                 and base <= n["start"] < base + len(text)]

        def leaf_nodes(leaf):
            lo, hi = base + cmap[leaf["start"]], base + cmap[min(leaf["end"], len(cmap) - 1)]
            return [n["node_id"] for n in local if n["start"] < hi and n["end"] > lo]

        def head(tree):
            if "left" not in tree:                    # a leaf may hold a clause plus equations: chain them
                ids = leaf_nodes(tree)
                edges.extend(_edge(a, b, "discourse", "within_leaf", "NN") for a, b in zip(ids, ids[1:]))
                return ids[0] if ids else None
            left, right = head(tree["left"]), head(tree["right"])
            if left is not None and right is not None:
                edges.append(_edge(min(left, right), max(left, right), "discourse",
                                   tree["relation"], tree["nuclearity"]))
            nucleus = right if tree["nuclearity"] == "SN" else left
            return nucleus if nucleus is not None else (left if left is not None else right)

        for tree in trees:
            head(tree)
    return _unique(edges)


def tool_edges(nodes):
    calls = {n["call_id"]: n["node_id"] for n in nodes if n["kind"] == "tool_call" and n["call_id"]}
    edges = []
    for out in (n for n in nodes if n["kind"] == "tool_output"):
        call = calls.get(out["call_id"])
        if call is not None and call < out["node_id"]:
            edges.append(_edge(call, out["node_id"], "tool", "tool_result"))
        later = [n for n in nodes if n["node_id"] > out["node_id"] and n["role"] == "assistant"
                 and n["part_id"] != out["part_id"]]
        if later:
            edges.append(_edge(out["node_id"], later[0]["node_id"], "tool", "tool_response"))
    return edges


def long_range_edges(nodes, embeddings, k_similar, threshold):
    """For each target-eligible node: the top-k earlier nodes from other parts by cosine
    above the threshold, plus the nearest earlier node from another part sharing an identifier."""
    unit = np.asarray(embeddings, dtype=float)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    identifiers = [set(IDENTIFIER.findall(n["text"] or "")) for n in nodes]
    edges = []
    for t in nodes:
        if t["kind"] not in TARGETS:
            continue
        earlier = [s["node_id"] for s in nodes if s["node_id"] < t["node_id"] and s["part_id"] != t["part_id"]]
        if not earlier:
            continue
        cosines = unit[earlier] @ unit[t["node_id"]]
        for i in np.argsort(-cosines, kind="stable")[:k_similar]:
            if cosines[i] >= threshold:
                edges.append(_edge(earlier[i], t["node_id"], "similarity", cosine=float(cosines[i])))
        shared = [s for s in earlier if identifiers[s] & identifiers[t["node_id"]]]
        if shared:
            edges.append(_edge(shared[-1], t["node_id"], "identifier",
                               cosine=float(unit[shared[-1]] @ unit[t["node_id"]])))
    return edges


def cap_parents(edges, max_parents):
    """Per target: all discourse/tool parents first, then long-range parents by cosine, up to max_parents."""
    by_target = {}
    for e in _unique(edges):
        by_target.setdefault(e["target_node_id"], []).append(e)
    kept = []
    for target in sorted(by_target):
        group = by_target[target]
        structural = [e for e in group if e["edge_source"] in STRUCTURAL]
        rest = sorted((e for e in group if e["edge_source"] not in STRUCTURAL),
                      key=lambda e: (-(e["cosine"] or 0.0), e["source_node_id"]))
        kept += (structural + rest)[:max_parents]
    return kept


def _unique(edges):
    """One edge per (source, target); structural sources win over long-range ones."""
    best = {}
    for e in edges:
        key = (e["source_node_id"], e["target_node_id"])
        if key not in best or RANK[e["edge_source"]] < RANK[best[key]["edge_source"]]:
            best[key] = e
    return [best[key] for key in sorted(best)]
