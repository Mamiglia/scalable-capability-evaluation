"""Chronological clause/tool labels for exported transcripts, from the IsaNLP RST parser.

    python scripts/glm_features/rst_labels.py <transcripts dir> --out <labels dir> [--shard i --n-shards n] [--device cuda:0]

Runs in the parser environment (isanlp_rst, pandas, pyarrow; no project imports). Reads every
<transcripts dir>/<bench>.parquet written by export_transcripts.py and writes
<labels dir>/shard-<i>.jsonl, one line per transcript:
  {"log", "item", "epoch", "labels": [[type, in_reasoning], ...], "failed_parts": n}
Types are protocols.transcript_features.features.CLAUSE_TYPES: a prose clause takes the RST
relation of its nearest discourse edge (parent first, then child), else continuation /
same_unit for links inside one unit, else root; fenced code, display maths, headings and
ANSWER lines are protected spans with their own type; tool calls are tool_inspect or tool_act
by a lexicon; tool outputs and later user turns are tool_output and feedback. Rules and
lexicon follow the earlier pilot (small-judge judge-dissect, scripts/discourse_network).
A part the parser rejects is dropped and counted in failed_parts.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

PARSER = {"hf_model_name": "tchewik/isanlp_rst_v3", "hf_model_version": "gumrrg"}
RELATIONS = {"elaboration", "joint", "organization", "restatement", "causal", "explanation", "contingency",
             "purpose", "adversative", "context", "evaluation", "attribution", "mode", "topic"}
STRUCTURAL = {"within_leaf": "continuation", "same-unit": "same_unit", "same_unit": "same_unit"}
PROTECTED = re.compile(r"```[\s\S]*?```|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")   # fenced code, display maths
HEADING = re.compile(r"^#{1,6} .*$", re.M)
ANSWER = re.compile(r"^ANSWER:.*$", re.M)
INLINE = re.compile(r"\\\(.*?\\\)|\$[^$\n]+\$|`[^`\n]+`")                     # inline maths/code stays in its clause
TOOL_TYPES = {
    "inspect": ["ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "sed -n", "less", "more", "stat", "file",
                "which", "pwd", "echo", "tree", "diff", "git status", "git log", "git diff", "git show", "view",
                "read_file", "list_dir", "search"],
    "act": ["submit", "sed -i", "rm", "mv", "cp", "touch", "mkdir", "tee", ">", ">>", "pip", "git apply",
            "git commit", "git checkout", "patch", "chmod", "create", "str_replace", "insert", "write_file",
            "edit_file", "apply_patch"],
}
PYTHON_WRITE_MARKERS = ["'w'", '"w"', "'a'", '"a"', "os.remove", "os.rename", "shutil.", "subprocess", "os.system",
                        ".write_text", ".unlink"]


def protected_spans(text):
    """(start, end, kind) of spans never split into clauses: code, equation, heading, answer."""
    spans = [(m.start(), m.end(), "code" if m.group().startswith("```") else "equation")
             for m in PROTECTED.finditer(text)]
    for pattern, kind in ((HEADING, "heading"), (ANSWER, "answer")):
        spans += [(m.start(), m.end(), kind) for m in pattern.finditer(text)
                  if not any(s <= m.start() < e for s, e, _ in spans)]
    return sorted(spans)


def mask_prose(text):
    """Text for the parser: protected spans become EQUATION, inline maths/code MATH, bold markers
    spaces. Returns (masked, cmap, protected); cmap[i] is the original offset of masked char i."""
    protected = protected_spans(text)
    spans = [(s, e, "EQUATION") for s, e, _ in protected]
    spans += [(m.start(), m.end(), "MATH") for m in INLINE.finditer(text)
              if not any(s <= m.start() < e for s, e, _ in spans)]
    out, cmap, cursor = [], [], 0
    for s, e, word in sorted(spans):
        out += [text[cursor:s], word]
        cmap += list(range(cursor, s)) + [s] * len(word)
        cursor = e
    out.append(text[cursor:])
    cmap += list(range(cursor, len(text) + 1))
    return "".join(out).replace("**", "  "), cmap, protected


def leaves_and_edges(tree, shift, leaves, edges):
    """Post-order walk of one isanlp DiscourseUnit: leaves get (start, end) in masked
    coordinates; every internal node yields (head_left, head_right, relation, nuclearity).
    Returns the head leaf index of `tree` (the nucleus child's head)."""
    if tree.left is None:
        leaves.append((tree.start + shift, tree.end + shift))
        return len(leaves) - 1
    left, right = leaves_and_edges(tree.left, shift, leaves, edges), leaves_and_edges(tree.right, shift, leaves, edges)
    edges.append((left, right, str(tree.relation).lower(), str(tree.nuclearity)))
    return right if tree.nuclearity == "SN" else left


def leaf_types(n_leaves, edges):
    """One type per leaf: nearest rhetorical parent edge (largest source), else nearest
    rhetorical child edge (smallest target), else the structural label, else root."""
    parent, child, s_parent, s_child = {}, {}, {}, {}
    for source, target, relation, _ in edges:
        name = relation.split("_")[0].split("-")[0] if relation not in STRUCTURAL else relation
        if name in RELATIONS:
            if source > parent.get(target, (-1,))[0]:
                parent[target] = (source, name)
            if target < child.get(source, (10**9,))[0]:
                child[source] = (target, name)
        elif relation in STRUCTURAL:
            if source > s_parent.get(target, (-1,))[0]:
                s_parent[target] = (source, STRUCTURAL[relation])
            if target < s_child.get(source, (10**9,))[0]:
                s_child[source] = (target, STRUCTURAL[relation])
    out = []
    for i in range(n_leaves):
        hit = parent.get(i) or child.get(i) or s_parent.get(i) or s_child.get(i)
        out.append(hit[1] if hit else "root")
    return out


def prose_labels(parser, text):
    """Ordered node types of one prose part: clause types from the parser interleaved with
    protected spans by position."""
    masked, cmap, protected = mask_prose(text)
    stripped = masked.strip()
    nodes = [(s, kind) for s, _, kind in protected]
    if stripped:
        lead = len(masked) - len(masked.lstrip())
        leaves, edges = [], []
        for tree in parser(stripped)["rst"]:
            leaves_and_edges(tree, lead, leaves, edges)
        types = leaf_types(len(leaves), edges)
        for (start, end), kind in zip(leaves, types):
            a, b = cmap[min(start, len(cmap) - 1)], cmap[min(end, len(cmap) - 1)]
            # a leaf whose text is only protected material adds no clause; otherwise one clause at its start
            if text[a:b].strip() and not any(s <= a and b <= e for s, e, _ in protected):
                inner = [x for s, e, _ in protected for x in (s, e) if a < x < b]
                cuts = [a] + inner + [b]
                for lo, hi in zip(cuts, cuts[1:]):
                    if text[lo:hi].strip() and not any(s <= lo and hi <= e for s, e, _ in protected):
                        nodes.append((lo, kind))
    return [kind for _, kind in sorted(nodes)]


def tool_type(function, content):
    """'inspect' (read-only) or 'act' (changes state). Python reads unless it writes or shells
    out; otherwise the first matching lexicon group wins and unmatched calls count as act."""
    text = f"{function or ''} {content}"
    if function == "python":
        return "act" if any(m in text for m in PYTHON_WRITE_MARKERS) else "inspect"
    for group in ("act", "inspect"):
        if any(_has_term(text, term) for term in TOOL_TYPES[group]):
            return group
    return "act"


def _has_term(text, term):
    if (term[0].isalnum() or term[0] == "_") and (term[-1].isalnum() or term[-1] == "_"):
        return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text) is not None
    return term in text


def transcript_labels(parser, parts):
    labels, failed = [], 0
    for part in parts:
        kind = part["kind"]
        if kind in ("text", "reasoning"):
            try:
                types = prose_labels(parser, part["text"])
            except Exception as error:  # parser rejects some inputs (e.g. no tokens); the part is dropped
                if type(error).__name__ == "OutOfMemoryError":  # GPU pressure, not this part: die and resume later
                    raise
                failed += 1
                continue
            labels += [[t, kind == "reasoning"] for t in types]
        elif kind == "tool_call":
            labels.append([f"tool_{tool_type(part.get('function'), part['text'])}", False])
        elif kind == "tool_output":
            labels.append(["tool_output", False])
        elif kind == "feedback":
            labels.append(["feedback", False])
    return labels, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cpu", help="cpu, cuda:<n> or mps")
    ap.add_argument("--limit", type=int, default=None, help="probe: parse only the first N transcripts of the shard")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(p) for p in sorted(args.transcripts.glob("*.parquet"))]
    table = pd.concat(frames, ignore_index=True).sort_values(["log", "item"]).reset_index(drop=True)
    table = table.iloc[args.shard::args.n_shards]
    if args.limit:
        table = table.head(args.limit)
    out = args.out / f"shard-{args.shard}.jsonl"
    done = set()
    if out.exists():
        with out.open() as f:
            done = {(json.loads(line)["log"], json.loads(line)["item"]) for line in f if line.strip()}
    todo = table[[(l, i) not in done for l, i in zip(table.log, table.item)]]
    print(f"shard {args.shard}/{args.n_shards}: {len(table)} transcripts, {len(done)} done, {len(todo)} to do", flush=True)
    if todo.empty:
        return
    import torch
    from isanlp_rst.parser import Parser
    parser = Parser(cuda_device=-1, **PARSER)
    if args.device != "cpu":
        device = torch.device(args.device)
        pred = parser.predictor
        pred._cuda_device = device
        pred.model.to(device)
        for module in pred.model.modules():
            for key, value in list(vars(module).items()):
                if isinstance(value, torch.device):
                    setattr(module, key, device)
    t0, chars = time.time(), 0
    with out.open("a") as f:
        for n, row in enumerate(todo.itertuples(), 1):
            parts = json.loads(row.parts)
            labels, failed = transcript_labels(parser, parts)
            chars += sum(len(p["text"]) for p in parts if p["kind"] in ("text", "reasoning"))
            f.write(json.dumps({"log": row.log, "item": row.item, "epoch": int(row.epoch), "labels": labels,
                                "failed_parts": failed}) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"  {n}/{len(todo)} {chars / dt:.0f} chars/s elapsed {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
