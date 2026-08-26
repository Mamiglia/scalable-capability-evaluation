"""Per-transcript feature blocks.

Block 1, typed topology: counts and shares of node types, RST relation shares, edge-source
shares, depth/branching/ancestry of the graph, and sequence motifs (inspect before act,
verification after act, error then correction, restatement chains).
Block 2, discovered types: a small Top-K SAE on node embeddings and one on edge vectors,
aggregated per transcript as mean activation and fraction of units active per feature.
"""
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NODE_TYPES = ["assistant_edu", "equation", "code", "heading", "answer", "tool_inspect", "tool_act",
              "output_inspect", "output_act", "user_feedback"]
EDGE_SOURCES = ["discourse", "tool", "similarity", "identifier"]
LONG_RANGE = 8                                     # parent more than this many nodes back


def node_type(n):
    """Node kind, with tool calls/outputs split by inspect/act."""
    if n["kind"] == "tool_call":
        return f"tool_{n['tool_type'] or 'act'}"
    if n["kind"] == "tool_output":
        return f"output_{n['tool_type'] or 'act'}"
    return n["kind"]


def _graph(n_nodes, edges):
    outgoing, incoming = {i: [] for i in range(n_nodes)}, {i: [] for i in range(n_nodes)}
    for e in edges:
        outgoing[e["source_node_id"]].append(e["target_node_id"])
        incoming[e["target_node_id"]].append(e["source_node_id"])
    return outgoing, incoming


def _ancestors(sink, incoming):
    seen, stack = {sink}, [sink]
    while stack:
        for s in incoming[stack.pop()]:
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def topology_block(nodes, edges, relation_names=None):
    """Block 1 for one transcript (nodes in node_id order, edges chronological)."""
    n, m = len(nodes), len(edges)
    outgoing, incoming = _graph(n, edges)
    depth = [0] * n                                  # longest path ending at each node; edges go forward
    for i in range(n):
        depth[i] = 1 + max((depth[s] for s in incoming[i]), default=-1)
    components, unseen = 0, set(range(n))
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            i = stack.pop()
            for j in outgoing[i] + incoming[i]:
                if j in unseen:
                    unseen.remove(j)
                    stack.append(j)
    sinks = [x["node_id"] for x in nodes if x["is_final_sink"]]
    ancestry = _ancestors(sinks[0], incoming) if sinks else set()
    types = Counter(node_type(x) for x in nodes)
    ancestry_types = Counter(node_type(nodes[i]) for i in ancestry)
    outputs = [x for x in nodes if x["kind"] == "tool_output"]
    relation_counts = Counter(e["relation"] for e in edges)
    relation_names = relation_names or sorted(r for r in relation_counts if r is not None)
    sources = Counter(e["edge_source"] for e in edges)
    distances = [e["target_node_id"] - e["source_node_id"] for e in edges]
    f = {"node_count": n, "edge_count": m,
         "longest_path": max(depth), "branch_nodes": sum(len(v) > 1 for v in outgoing.values()),
         "convergence_nodes": sum(len(v) > 1 for v in incoming.values()), "weak_components": components,
         "disconnected_share": sum(not outgoing[i] and not incoming[i] for i in range(n)) / n,
         "long_range_edge_share": sum(d > LONG_RANGE for d in distances) / max(m, 1),
         "parent_distance_mean": float(np.mean(distances)) if distances else 0.0,
         "parent_distance_max": max(distances, default=0),
         "final_ancestor_share": len(ancestry) / n, "abandoned_share": 1 - len(ancestry) / n,
         "question_edges": len(outgoing[0]),
         "tool_feedback_edges": sum(len(outgoing[x["node_id"]]) for x in nodes
                                    if x["kind"] in {"tool_output", "user_feedback"}),
         "output_error_share": (sum(x["outcome"] == "error" for x in outputs) / len(outputs)) if outputs else 0.0,
         "marker_backtracking": sum(bool(x.get("backtracking_marker")) for x in nodes),
         "marker_verification": sum(bool(x.get("verification_marker")) for x in nodes)}
    f |= {f"share_{t}": types[t] / n for t in NODE_TYPES}
    f |= {f"ancestry_share_{t}": ancestry_types[t] / n for t in NODE_TYPES}
    f |= {f"relation_{r}": relation_counts[r] / max(m, 1) for r in relation_names}
    f |= {f"edge_source_{s}": sources[s] / max(m, 1) for s in EDGE_SOURCES}
    f |= _motifs(nodes, edges)
    return {f"topo_{k}": float(v) for k, v in f.items()}


def _motifs(nodes, edges):
    """Sequence motifs along node order; each is a share of the anchoring events."""
    acts = [x for x in nodes if node_type(x) == "tool_act"]
    inspects = [x["node_id"] for x in nodes if node_type(x) == "tool_inspect"]
    errors = [x for x in nodes if x["kind"] == "tool_output" and x["outcome"] == "error"]

    def response(x):                                 # assistant nodes of the next assistant part (at most 20)
        later = [y for y in nodes[x["node_id"] + 1:] if y["role"] == "assistant" and y["part_id"] != x["part_id"]]
        return [y for y in later if y["part_id"] == later[0]["part_id"]][:20] if later else []

    def share(events, hit):
        return sum(hit(x) for x in events) / len(events) if events else 0.0

    nxt = {e["source_node_id"]: e["target_node_id"] for e in edges if e["relation"] == "restatement"}
    chain = 0
    for start in nxt:                                # longest run of consecutive restatement edges
        length, cur = 0, start
        while cur in nxt:
            cur, length = nxt[cur], length + 1
        chain = max(chain, length)
    return {"act_after_inspect": share(acts, lambda x: any(i < x["node_id"] for i in inspects)),
            "verification_after_act": share(acts, lambda x: any(y.get("verification_marker") for y in response(x))),
            "error_then_correction": share(errors, lambda x: any(y.get("backtracking_marker") for y in response(x))),
            "restatement_chain_max": chain}


# ---------------------------------------------------------------- block 2: SAEs

class TopKSAE:
    """Tied-weight Top-K sparse autoencoder: z = topk(relu(x W^T)), x_hat = z W, rows of W unit norm."""

    def __init__(self, decoder, top_k):
        self.decoder, self.top_k = decoder, top_k

    @classmethod
    def fit(cls, x, width, top_k, iterations=20, batch_size=256, seed=0, learning_rate=1e-3):
        torch.manual_seed(seed)
        data = torch.as_tensor(np.asarray(x, dtype=np.float32))
        w = torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(width, data.shape[1]), dim=1))
        opt = torch.optim.Adam([w], lr=learning_rate)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(iterations):
            for idx in torch.randperm(len(data), generator=gen).split(batch_size):
                batch = data[idx]
                scores = torch.relu(batch @ w.T)
                top = torch.topk(scores, top_k, dim=1)
                z = torch.zeros_like(scores).scatter(1, top.indices, top.values)
                loss = torch.mean((batch - z @ w) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    w.copy_(torch.nn.functional.normalize(w, dim=1))
        return cls(w.detach().numpy(), top_k)

    def transform(self, x):
        scores = np.maximum(np.asarray(x, dtype=np.float32) @ self.decoder.T, 0.0)
        keep = np.argpartition(scores, -self.top_k, axis=1)[:, -self.top_k:]
        z = np.zeros_like(scores)
        rows = np.arange(len(scores))[:, None]
        z[rows, keep] = scores[rows, keep]
        return z


class EdgeSpace:
    """Edge vector = PCA of standardised [s, t, t - s, s * t] of the two node embeddings."""

    def __init__(self, scaler, pca):
        self.scaler, self.pca = scaler, pca

    @staticmethod
    def pair_features(embeddings, edges):
        s = embeddings[[e["source_node_id"] for e in edges]]
        t = embeddings[[e["target_node_id"] for e in edges]]
        return np.concatenate((s, t, t - s, s * t), axis=1)

    @classmethod
    def fit(cls, embeddings, edges, components, seed=0):
        raw = cls.pair_features(embeddings, edges)
        scaler = StandardScaler().fit(raw)
        return cls(scaler, PCA(n_components=components, random_state=seed).fit(scaler.transform(raw)))

    def transform(self, embeddings, edges):
        return self.pca.transform(self.scaler.transform(self.pair_features(embeddings, edges))).astype(np.float32)


def sae_block(activations, unit_owner, n_transcripts, prefix, min_support, support_mask=None):
    """Per-transcript mean activation and fraction of units active, for features active on at
    least min_support units. unit_owner[i] is the transcript index of unit i."""
    z, owner = np.asarray(activations), np.asarray(unit_owner)
    support = z if support_mask is None else z[np.asarray(support_mask)]
    retained = [j for j in range(z.shape[1]) if (support[:, j] > 0).sum() >= min_support]
    counts = np.maximum(np.bincount(owner, minlength=n_transcripts), 1)
    sums, active = np.zeros((n_transcripts, z.shape[1])), np.zeros((n_transcripts, z.shape[1]))
    np.add.at(sums, owner, z)
    np.add.at(active, owner, z > 0)
    frame = {}
    for j in retained:
        frame[f"{prefix}_f{j:02d}_mean"] = sums[:, j] / counts
        frame[f"{prefix}_f{j:02d}_active"] = active[:, j] / counts
    return pd.DataFrame(frame)
