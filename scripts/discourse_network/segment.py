"""Clause boundaries and RST trees for every assistant text part, from the pinned
isanlp_rst parser. Runs in the parser venv (configs/workstream_c.yaml: segmenter.python);
only main() imports isanlp_rst, so the pure functions are testable in the project venv.

  <segmenter.python> scripts/discourse_network/segment.py --shard 0 --n-shards 6

Writes <out_dir>/segments/shard-<i>.clauses.parquet (one row per clause, transcript
offsets) and shard-<i>.trees.parquet (one RST tree per assistant text part, as JSON in
masked coordinates). A shard whose clauses file exists is skipped.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discourse_network.nodes import leaf_to_clauses, mask_prose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
TOKEN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def tree_dict(node, shift=0):
    """isanlp DiscourseUnit -> nested dict; leaves keep only their span."""
    out = {"start": node.start + shift, "end": node.end + shift}
    if node.left is not None:
        out |= {"relation": node.relation, "nuclearity": node.nuclearity,
                "left": tree_dict(node.left, shift), "right": tree_dict(node.right, shift)}
    return out


def leaves(node):
    return [node] if node.left is None else leaves(node.left) + leaves(node.right)


def normalise_tree_spans(trees, text):
    """Snap ordered parser leaves to a complete partition of whole text tokens."""
    leaf_nodes = []

    def collect(node):
        if "left" not in node:
            leaf_nodes.append(node)
        else:
            collect(node["left"])
            collect(node["right"])

    for tree in trees:
        collect(tree)
    tokens = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
    if len(leaf_nodes) > len(tokens):
        raise ValueError(f"parser returned {len(leaf_nodes)} leaves for {len(tokens)} tokens")
    previous, cuts = 0, []
    for i, (left, right) in enumerate(zip(leaf_nodes, leaf_nodes[1:]), 1):
        target = (left["end"] + right["start"]) / 2
        last = len(tokens) - (len(leaf_nodes) - i)
        cut = min(range(previous + 1, last + 1),
                  key=lambda j: abs((tokens[j - 1][1] + tokens[j][0]) / 2 - target))
        cuts.append(cut)
        previous = cut
    for leaf, (start, end) in zip(leaf_nodes, zip([0] + cuts, cuts + [len(tokens)])):
        leaf["start"], leaf["end"] = tokens[start][0], tokens[end - 1][1]

    def update(node):
        if "left" in node:
            update(node["left"])
            update(node["right"])
            node["start"], node["end"] = node["left"]["start"], node["right"]["end"]

    for tree in trees:
        update(tree)
    return trees


def validate_spans(text, clauses, protected):
    """Require clauses and protected spans to cover source text once, except whitespace."""
    cursor = 0
    for start, end in sorted(clauses + [(s, e) for s, e, _ in protected]):
        if start < cursor:
            raise ValueError(f"overlapping segment spans at {start}:{cursor}")
        if text[cursor:start].replace("**", "").strip():
            raise ValueError(f"uncovered segment text at {cursor}:{start}")
        if ((start and text[start - 1].isalnum() and text[start].isalnum()) or
                (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())):
            raise ValueError(f"segment boundary inside a word at {start}:{end}")
        cursor = end
    if text[cursor:].replace("**", "").strip():
        raise ValueError(f"uncovered segment text at {cursor}:{len(text)}")


def segment_text(parser, text):
    """Clause spans (offsets into `text`) and tree dicts for one assistant text part."""
    masked, cmap, protected = mask_prose(text)
    stripped = masked.strip()
    if not stripped:
        return [], []
    lead = len(masked) - len(masked.lstrip())     # parser sees the stripped text; shift offsets back
    clauses, trees = [], []
    for tree in parser(stripped)["rst"]:
        trees.append(tree_dict(tree, lead))
    normalise_tree_spans(trees, masked)
    for tree in trees:
        for leaf in _dict_leaves(tree):
            clauses += leaf_to_clauses(leaf["start"], leaf["end"], cmap, text, protected)
    clauses = sorted(set(clauses))
    validate_spans(text, clauses, protected)
    return clauses, trees


def _dict_leaves(node):
    return [node] if "left" not in node else _dict_leaves(node["left"]) + _dict_leaves(node["right"])


def segment_rows(rows, parser):
    """Clause and tree tables for store rows (dicts with A's keys and transcript_parts_json)."""
    clause_rows, tree_rows = [], []
    for row in rows:
        keys = {k: row[k] for k in KEYS}
        for part_id, part in enumerate(json.loads(row["transcript_parts_json"])):
            if part["role"] != "assistant":
                continue
            for subpart_id, sp in enumerate(part.get("subparts", [])):
                if sp["kind"] != "content":
                    continue
                clauses, trees = segment_text(parser, sp["content"])
                where = {"part_id": part_id, "subpart_id": subpart_id}
                clause_rows += [keys | where | {"start": sp["start"] + a, "end": sp["start"] + b}
                                for a, b in clauses]
                tree_rows += [keys | where | {"tree_json": json.dumps(tree)} for tree in trees]
    return pd.DataFrame(clause_rows), pd.DataFrame(tree_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "workstream_c.yaml")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"] / "segments"
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"shard-{args.shard}.clauses.parquet"
    if done.exists():
        print(f"already done: {done}")
        return
    store = pd.read_parquet(ROOT / cfg["freeze"] / "transcript_store.parquet")
    store = store[store.correct & store.is_canonical].sort_values(KEYS).reset_index(drop=True)
    store = store.iloc[args.shard::args.n_shards]
    from isanlp_rst.parser import Parser
    parser = Parser(hf_model_name=cfg["segmenter"]["model"], hf_model_version=cfg["segmenter"]["version"],
                    cuda_device=-1)
    clauses, trees = segment_rows(store.to_dict("records"), parser)
    trees.to_parquet(out / f"shard-{args.shard}.trees.parquet", index=False)
    clauses.to_parquet(done.with_suffix(".tmp.parquet"), index=False)
    done.with_suffix(".tmp.parquet").rename(done)          # written last: its presence means the shard is done
    print(f"shard {args.shard}: {len(store)} transcripts, {len(clauses)} clauses, {len(trees)} trees")


if __name__ == "__main__":
    main()
