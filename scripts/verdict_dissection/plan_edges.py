"""Plan-realisation edges: link each item of a stated plan to the later step that carries it out.

    uv run python scripts/verdict_dissection/plan_edges.py [--show 6] [--seed 0]

A plan is a run of two or more list-item clauses (- / * / 1. / Step 1) in the transcript's first
assistant message, before any tool call (agentic) or within the first third of the nodes
(single-message), announced by a cue ("Plan:", "Steps:", "We will", "Let's", "To solve" ...) in the
three clauses before the list. Each item is the marker clause plus the clauses that follow it up to the next
marker. Only transcripts that use tools get plan edges: each item is matched to the first later tool call whose
command class (list / read / search / test / run / edit / submit) matches the item's action words,
scanning forward in plan order; with no class match, the later call with the highest embedding cosine,
flagged as unmatched (overlap 0).
Writes <network>/plan_edges.parquet (transcript_key, source_node_id = plan item, target_node_id,
cosine, overlap, score) and <features_root>/raw/clause-plan.parquet with per-transcript features:
plan_items, plan_present, plan_executed_share (class-matched calls), plan_order_concordance (Kendall tau
between plan order and execution order), plan_unplanned_share (later tool calls no item maps to),
plan_position (where the plan sits, as a share of nodes). --show prints examples for checking.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from features_transitions import KEYS, NETWORK   # noqa: E402

MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)]|Step\s*\d+[.:)]?)\s+", re.I)
PLAN_CUE = re.compile(r"\b(?:plan|steps|approach|strategy|outline|we will|i will|i'll|to solve|here'?s how)\b", re.I)
MAX_ITEM_WORDS = 30                             # longer list items are exposition, not plan steps
CUE_WINDOW = 3                                  # clauses before the first list item that may carry the cue
WORD = re.compile(r"[a-z][a-z0-9_.\-]+")
STOP = {"the", "and", "for", "with", "that", "this", "then", "into", "from", "any", "all", "its", "are", "was", "will",
        "use", "using", "find", "check", "look", "see", "get", "make", "need", "needed", "provided", "given", "step",
        "first", "next", "let", "lets", "now", "should", "could", "would", "have", "has", "not", "can"}
EXECUTED_SCORE = 0.5
ACTIONS = {                                     # plan wording -> command class; a step is "executed" by a call of that class
    "list": re.compile(r"\b(?:list|explore|look around|what files|directory contents|see (?:what|available))\b", re.I),
    "read": re.compile(r"\b(?:read|inspect|view|examine|open|look at|review|understand|check (?:the )?(?:file|script|code|source)|"
                       r"cat\b|contents?)\b", re.I),
    "search": re.compile(r"\b(?:search|grep|find|locate|look for)\b", re.I),
    "test": re.compile(r"\b(?:run (?:the )?tests?|test suite|pytest|failing tests?|re-?run tests?)\b", re.I),
    "run": re.compile(r"\b(?:run|execute|invoke|try|decompile|disassemble|decrypt|decode|brute.?force|compute|calculate|convert)\b", re.I),
    "edit": re.compile(r"\b(?:modify|edit|fix|patch|change|update|implement|apply|write|add|adjust|refactor)\b", re.I),
    "submit": re.compile(r"\b(?:submit|answer|flag)\b", re.I),
}
COMMAND_CLASS = {"list": r"\b(?:ls|tree|find \.? ?-maxdepth)\b", "read": r"\b(?:cat|sed -n|head|tail|less|more|file|strings|hexdump|xxd|objdump|nl)\b",
                 "search": r"\b(?:grep|rg|find|ag|locate)\b", "test": r"\b(?:pytest|runtests\.py|tox|unittest|npm test|make test|cargo test|go test)\b",
                 "run": r"\b(?:python3?|bash|sh|gdb|r2|radare2|binwalk|gunzip|tar|unzip|base64|openssl|xxd -r|nc|curl|wget|chmod|\./)\b",
                 "edit": r"\b(?:sed -i|applypatch|patch|tee|cat >|echo >|Path\(|write_text|open\(.*['\"]w|git apply|>>)\b",
                 "submit": r"^\[Tool Call: submit"}
COMMAND_CLASS = {k: re.compile(v, re.M) for k, v in COMMAND_CLASS.items()}
CALL_HEAD = re.compile(r"^\[Tool Call: (\w+)\(\{'(?:command|code|answer|cmd)': ", re.S)
WRITES_FILE = re.compile(r"write_text|open\([^)]*['\"][wa]|\.write\(|shutil\.|os\.replace|Path\([^)]*\)\.(?:write|unlink)")


def words(text):
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP}


def plan_items(group):
    """Plan item node ids and their texts for one transcript (nodes in node_id order), or []."""
    assistant = group[group.kind == "assistant_edu"]
    if assistant.empty:
        return []
    first_part = assistant.part_id.iloc[0]
    tools = group[group.kind == "tool_call"]
    limit = tools.node_id.iloc[0] if len(tools) else len(group) / 3
    block = assistant[(assistant.part_id == first_part) & (assistant.node_id < limit)]
    markers = block[block.text.fillna("").str.match(MARKER)]
    if len(markers) < 2:
        return []
    before = block[block.node_id < markers.node_id.iloc[0]].tail(CUE_WINDOW)
    if not before.text.fillna("").str.contains(PLAN_CUE).any():
        return []
    items, ids = [], markers.node_id.tolist()
    for start, end in zip(ids, ids[1:] + [block.node_id.iloc[-1] + 1]):
        text = MARKER.sub("", " ".join(block[(block.node_id >= start) & (block.node_id < end)].text.fillna("")), count=1)
        if len(text.split()) <= MAX_ITEM_WORDS:
            items.append((start, text))
    return items if len(items) >= 2 else []


def action_classes(text):
    return {name for name, pattern in ACTIONS.items() if pattern.search(text or "")}


def command_classes(text):
    """Classes of one tool call, from its tool name and the command/code payload (the wrapper text is ignored)."""
    head = CALL_HEAD.match(text or "")
    tool, payload = (head.group(1), (text or "")[head.end():]) if head else ("", text or "")
    if tool == "submit":
        return {"submit"}
    if tool == "python":
        return {"edit"} if WRITES_FILE.search(payload) else {"run"}
    return {name for name, pattern in COMMAND_CLASS.items() if name != "submit" and pattern.search(payload)}


def match_plan(group, embeddings, items):
    """Each plan item -> the first later tool call whose command class matches the item's action words,
    scanning forward from the previous item's match (plans are executed in order); if none matches,
    the later call with the highest cosine, flagged by overlap 0. Score = 1 (class match) + cosine."""
    last_plan = items[-1][0]
    calls = group[(group.node_id > last_plan) & (group.kind == "tool_call")]
    if calls.empty:
        return []
    call_ids = calls.node_id.to_numpy()
    call_classes = [command_classes(t) for t in calls.text]
    call_vec = embeddings[calls.index.to_numpy()]
    rows, cursor = [], 0
    for node_id, text in items:
        member = group[(group.node_id >= node_id) & (group.node_id <= last_plan) & (group.kind == "assistant_edu")]
        vec = embeddings[member.index.to_numpy()].mean(axis=0)
        vec /= max(np.linalg.norm(vec), 1e-12)
        cosine = call_vec @ vec
        wanted = action_classes(text)
        hits = [i for i in range(cursor, len(call_ids)) if wanted & call_classes[i]]
        if hits:
            best, matched = hits[0], 1.0
            cursor = best + 1
        else:
            best, matched = int(np.argmax(cosine)), 0.0
        rows.append({"source_node_id": node_id, "target_node_id": int(call_ids[best]), "cosine": float(cosine[best]),
                     "overlap": matched, "score": matched + float(cosine[best]),
                     "wanted": ",".join(sorted(wanted)), "call": ",".join(sorted(call_classes[best]))})
    return rows


def plan_features(group, edges):
    n = len(group)
    if not edges:
        return {"plan_items": 0, "plan_present": 0, "plan_executed_share": 0.0, "plan_order_concordance": 0.0,
                "plan_unplanned_share": 0.0, "plan_position": 0.0}
    executed = [e for e in edges if e["overlap"] >= 1.0]        # a class-matched call
    targets = [e["target_node_id"] for e in edges]
    tau = kendalltau(range(len(targets)), targets).statistic if len(set(targets)) > 1 else 0.0
    later_tools = group[(group.kind == "tool_call") & (group.node_id > edges[-1]["source_node_id"])].node_id
    unplanned = 1 - len(set(targets) & set(later_tools)) / len(later_tools) if len(later_tools) else 0.0
    return {"plan_items": len(edges), "plan_present": 1, "plan_executed_share": len(executed) / len(edges),
            "plan_order_concordance": 0.0 if np.isnan(tau) else float(tau), "plan_unplanned_share": float(unplanned),
            "plan_position": edges[0]["source_node_id"] / max(n, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", type=int, default=0, help="print this many transcripts with plans per benchmark, for checking")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    nodes = pd.read_parquet(NETWORK / "nodes.parquet", columns=KEYS + ["transcript_key", "node_id", "part_id", "kind", "text"])
    embeddings = np.load(NETWORK / "node_embeddings.npy", mmap_mode="r")      # rows aligned with nodes.parquet
    assert len(embeddings) == len(nodes)
    edge_rows, feature_rows, shown = [], [], {}
    keys = nodes.transcript_key.unique()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(keys) if args.show else keys
    by_key = nodes.groupby("transcript_key", sort=False)
    for key in order:
        group = by_key.get_group(key).sort_values("node_id")
        items = plan_items(group)
        edges = match_plan(group, embeddings, items) if items else []
        edge_rows += [{"transcript_key": key} | e for e in edges]
        feature_rows.append(dict(zip(KEYS, group.iloc[0][KEYS])) | plan_features(group, edges))
        bench = group.iloc[0]["bench"]
        if args.show and edges and shown.get(bench, 0) < args.show:
            shown[bench] = shown.get(bench, 0) + 1
            head = group.iloc[0]
            text = group.set_index("node_id").text.fillna("")
            print("=" * 110 + f"\n{head['bench']} | {head['model']} | item {head['item']} | {len(group)} nodes")
            for (node_id, item), e in zip(items, edges):
                print(f"  plan {node_id:3d}: {item[:70]!r}  [{e['wanted'] or '?'}]\n   -> {e['target_node_id']:3d} "
                      f"{'MATCH' if e['overlap'] else 'nearest'} [{e['call'] or '?'}] cos {e['cosine']:.2f} | {text[e['target_node_id']][:90]!r}")
            print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in plan_features(group, edges).items()})
        if args.show and len(shown) >= 6 and all(v >= args.show for v in shown.values()):
            return
    edges = pd.DataFrame(edge_rows)
    edges.to_parquet(NETWORK / "plan_edges.parquet", index=False)
    table = pd.DataFrame(feature_rows)
    table.to_parquet(Path(config["features_root"]) / "raw/clause-plan.parquet", index=False)
    print(f"{len(table)} transcripts, {int(table.plan_present.sum())} with a plan, {len(edges)} plan edges")
    print(table.groupby("bench")[["plan_present", "plan_items", "plan_executed_share", "plan_order_concordance", "plan_unplanned_share"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
