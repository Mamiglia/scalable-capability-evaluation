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
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discourse_network.nodes import leaf_to_clauses, mask_prose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]


def tree_dict(node, shift=0):
    """isanlp DiscourseUnit -> nested dict; leaves keep only their span."""
    out = {"start": node.start + shift, "end": node.end + shift}
    if node.left is not None:
        out |= {"relation": node.relation, "nuclearity": node.nuclearity,
                "left": tree_dict(node.left, shift), "right": tree_dict(node.right, shift)}
    return out


def leaves(node):
    return [node] if node.left is None else leaves(node.left) + leaves(node.right)


def segment_text(parser, text):
    """Clause spans (offsets into `text`) and tree dicts for one assistant text part."""
    masked, cmap, protected = mask_prose(text)
    stripped = masked.strip()
    if not stripped:
        return [], []
    lead = len(masked) - len(masked.lstrip())     # parser sees the stripped text; shift offsets back
    clauses, trees = [], []
    for tree in parser(stripped)["rst"]:
        for leaf in leaves(tree):
            clauses += leaf_to_clauses(leaf.start + lead, leaf.end + lead, cmap, text, protected)
        trees.append(tree_dict(tree, lead))
    return sorted(set(clauses)), trees


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
