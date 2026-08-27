"""Clause-type transition features: 1-step and 2-step transition shares over the clause network's nodes.

    uv run python scripts/verdict_dissection/features_transitions.py

Each node gets a type: a prose clause is typed by its nearest rhetorical parser relation (elaboration,
joint, organization, restatement, causal, explanation, contingency, purpose, adversative, context,
evaluation, attribution, mode, topic), else `continuation` / `same_unit` for links inside one discourse
unit, else `root`; equations, headings, code and
the answer keep their kind; tool calls split inspect/act, tool outputs ok/error. In reading order,
the 1-step matrix counts (type_i, type_i+1) and the 2-step matrix (type_i, type_i+2), each row
normalised by the transcript's number of transitions. Columns: t1__<from>__<to>, t2__<from>__<to>,
plus n_nodes. Writes <features_root>/raw/clause-transitions.parquet keyed like the other raw tables.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
NETWORK = ROOT / "judge_results/verdict_dissection/clause_network_v2_sim070"
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
RELATIONS = ["elaboration", "joint", "organization", "restatement", "causal", "explanation", "contingency",
             "purpose", "adversative", "context", "evaluation", "attribution", "mode", "topic"]
STRUCTURAL = {"within_leaf": "continuation", "same-unit": "same_unit"}     # links inside one discourse unit, not rhetorical
CLAUSE_TYPES = RELATIONS + list(STRUCTURAL.values()) + ["root"]
OTHER_TYPES = ["equation", "heading", "answer", "code", "tool_inspect", "tool_act", "output_ok", "output_error", "feedback"]
TYPES = CLAUSE_TYPES + OTHER_TYPES


def node_types(nodes, edges):
    """Type per node (index aligned with `nodes`), from kind, tool fields and the discourse edges.

    A prose clause takes the relation of its nearest discourse parent; if that is structural
    (within_leaf / same-unit) or missing, the nearest rhetorical relation among all its discourse
    edges (as target, then as source); else the structural label; else `root`."""
    discourse = edges[edges.edge_source == "discourse"]
    index = pd.MultiIndex.from_arrays([nodes.transcript_key, nodes.node_id])
    rhetorical = discourse[discourse.relation.isin(RELATIONS)]
    structural = discourse[discourse.relation.isin(STRUCTURAL)]

    def nearest(frame, own, other, keep):
        return (frame.sort_values(other).drop_duplicates(["transcript_key", own], keep=keep)
                .set_index(["transcript_key", own]).relation.reindex(index))

    parent = nearest(rhetorical, "target_node_id", "source_node_id", "last")
    child = nearest(rhetorical, "source_node_id", "target_node_id", "first")
    structural_parent = nearest(structural, "target_node_id", "source_node_id", "last")
    structural_child = nearest(structural, "source_node_id", "target_node_id", "first")
    relation = parent.fillna(child).fillna(structural_parent.map(STRUCTURAL)).fillna(structural_child.map(STRUCTURAL)).to_numpy()
    kind, tool = nodes.kind.to_numpy(), nodes.tool_type.fillna("act").to_numpy()
    out = np.where(pd.isna(relation), "root", relation)
    out = np.where(kind == "assistant_edu", out, kind)
    out = np.where(kind == "tool_call", np.char.add("tool_", tool.astype(str)), out)
    out = np.where(kind == "tool_output", np.where(nodes.outcome.to_numpy() == "error", "output_error", "output_ok"), out)
    out = np.where(kind == "user_feedback", "feedback", out)
    return out


def transition_shares(types):
    """1-step and 2-step transition shares for one transcript's type sequence (question excluded)."""
    seq = [t for t in types if t != "question"]
    one, two = Counter(zip(seq, seq[1:])), Counter(zip(seq, seq[2:]))
    row = {"n_nodes": len(seq)}
    row |= {f"t1__{a}__{b}": one[(a, b)] / max(len(seq) - 1, 1) for a in TYPES for b in TYPES}
    row |= {f"t2__{a}__{b}": two[(a, b)] / max(len(seq) - 2, 1) for a in TYPES for b in TYPES}
    return row


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    nodes = pd.read_parquet(NETWORK / "nodes.parquet", columns=KEYS + ["transcript_key", "node_id", "kind", "tool_type", "outcome"])
    edges = pd.read_parquet(NETWORK / "edges.parquet", columns=["transcript_key", "source_node_id", "target_node_id", "edge_source", "relation"])
    nodes = nodes.sort_values(["transcript_key", "node_id"]).reset_index(drop=True)
    nodes["type"] = node_types(nodes, edges)
    rows = []
    for key, group in nodes.groupby("transcript_key", sort=False):
        rows.append(dict(zip(KEYS, group.iloc[0][KEYS])) | transition_shares(group.type.tolist()))
    table = pd.DataFrame(rows)
    cells = [c for c in table.columns if c.startswith("t")]
    prevalence = (table[cells] > 0).mean()
    print(f"{len(table)} transcripts, {len(cells)} cells, {int((prevalence >= 0.05).sum())} with share > 0 in >= 5% of transcripts")
    print(prevalence.sort_values(ascending=False).head(25).round(3).to_string())
    table.to_parquet(features_root / "raw/clause-transitions.parquet", index=False)


if __name__ == "__main__":
    main()
