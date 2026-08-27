"""Side-by-side viewer: transcript text with each clause-network node marked, next to the graph.

    uv run python scripts/verdict_dissection/clause_graph_viewer.py [--per-bench 2] [--max-nodes 60] [--seed 1]

Writes judge_results/verdict_dissection/clause_network_v2_sim070/graph_viewer.html. Left: the transcript
with every node's span wrapped, numbered and coloured by type. Right: the nodes in reading order with
edges drawn as arcs (black = discourse parser, blue = tool flow, grey dashed = embedding similarity,
green = shared identifier), labelled with the relation. Hover a node on either side to highlight it and
its edges. Under both, a collapsible panel with every graph parameter of that transcript and the
percentile it sits at among its benchmark's transcripts. Self-contained, no libraries.
"""
import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from features_transitions import node_types, NETWORK, KEYS   # noqa: E402
FREEZE = ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66"
FEATURES = Path("/Volumes/T7/judge-dissection/transcript-features/raw")
FEATURE_GROUPS = [("Topology", "clause-network", lambda c: c.startswith("topo_aug_")),
                  ("Graph descriptors", "clause-descriptors", lambda c: True),
                  ("Shapes and motifs", "clause-motifs", lambda c: not c.startswith("wl__")),
                  ("Plan", "clause-plan", lambda c: True),
                  ("Exposition", "exposition", lambda c: True)]
OUT = NETWORK / "graph_viewer.html"
COLOURS = {"equation": "#c44", "heading": "#a0a", "answer": "#080", "code": "#886", "tool_inspect": "#06c", "tool_act": "#039",
           "output_ok": "#69c", "output_error": "#e60", "feedback": "#c90", "question": "#666"}
CLAUSE_COLOUR = "#333"
EDGE_STYLE = {"discourse": ("#222", ""), "tool": ("#06c", ""), "similarity": ("#999", "4 3"), "identifier": ("#2a2", ""),
              "plan": ("#d0a000", "6 2")}
ROW = 22


def within_bench_percentiles(frame, columns):
    """Each column ranked 0–1 inside its own benchmark, the three AIME years pooled into one."""
    bench = frame.bench.mask(frame.bench.str.startswith("aime"), "aime")
    return frame[columns].astype(float).groupby(bench.to_numpy()).rank(pct=True)


def feature_panels(key_map):
    """Per feature group: (heading, columns, values, within-benchmark percentiles), indexed by transcript_key."""
    panels = []
    for heading, table, keep in FEATURE_GROUPS:
        path = FEATURES / f"{table}.parquet"
        columns = [c for c in pq.read_schema(path).names if c not in KEYS and keep(c)]
        frame = pd.read_parquet(path, columns=KEYS + columns)
        values = frame[columns].astype(float)
        percentiles = within_bench_percentiles(frame, columns)
        key = frame.merge(key_map, on=KEYS, how="left").transcript_key.to_numpy()
        panels.append((heading, columns, values.set_index(key), percentiles.set_index(key)))
    return panels


def parameter_of(panels, key, column):
    """One feature's value for one transcript, whichever group it lives in."""
    for _, columns, values, _ in panels:
        if column in columns:
            return values.loc[key, column]
    return float("nan")


def show(value):
    return "–" if pd.isna(value) else f"{value:.0f}" if float(value).is_integer() else f"{value:.3g}"


def parameters_html(panels, key):
    """Collapsible panel: every parameter's value beside a bar for its percentile within the benchmark."""
    blocks = []
    for heading, columns, values, percentiles in panels:
        value_row, percentile_row = values.loc[key], percentiles.loc[key]
        rows = []
        for column in columns:
            percentile = percentile_row[column]
            place = "" if pd.isna(percentile) else f"{percentile * 100:.0f}"
            name = column.removeprefix("topo_aug_").replace("_", " ")
            rows.append(f'<div class="frow" title="{html.escape(column)}"><span class="fname">{html.escape(name)}</span>'
                        f'<span class="fval">{show(value_row[column])}</span>'
                        f'<span class="fbar"><i style="width:{place or 0}%"></i></span><span class="fpct">{place}</span></div>')
        blocks.append(f'<h3>{heading}</h3><div class="rows">{"".join(rows)}</div>')
    return ('<details class="params"><summary>graph parameters — each value with the percentile it sits at among '
            'this benchmark\'s transcripts (bar; tick = median, tied values sit mid-block)</summary>'
            f'{"".join(blocks)}</details>')


def summary_line(group, edge_rows, panels, key):
    """Nodes, edges by source, longest path and the plan, for the section header."""
    counts = edge_rows.edge_source.value_counts()
    by_source = ", ".join(f"{count} {source}" for source, count in counts.items())
    bits = [f"{len(group)} nodes", f"{len(edge_rows)} edges ({by_source})",
            f"longest path {show(parameter_of(panels, key, 'topo_aug_longest_path'))}"]
    plan_items = parameter_of(panels, key, "plan_items")
    if plan_items and not pd.isna(plan_items):
        bits.append(f"{show(plan_items)} plan items, {parameter_of(panels, key, 'plan_executed_share') * 100:.0f}% executed")
    return " · ".join(bits)


def render(transcript, group, edge_rows, title, summary, parameters):
    nodes = group.sort_values("node_id")
    spans = [(int(r.start), int(r.end), int(r.node_id), r.type) for r in nodes.itertuples() if pd.notna(r.start)]
    parts, cursor = [], 0
    for start, end, node_id, kind in sorted(spans):
        if start < cursor:                        # nested or overlapping span: mark it inline without slicing
            continue
        parts.append(html.escape(transcript[cursor:start]))
        colour = COLOURS.get(kind, CLAUSE_COLOUR)
        parts.append(f'<span class="node" data-id="{node_id}" style="border-color:{colour}">'
                     f'<b style="background:{colour}">{node_id}</b>{html.escape(transcript[start:end])}</span>')
        cursor = end
    parts.append(html.escape(transcript[cursor:]))
    text_html = "".join(parts)
    n = len(nodes)
    height = ROW * (n + 1)
    spine = 230                                   # arcs go to the left of the spine, labels to the right
    svg = [f'<svg width="620" height="{height}" data-n="{n}">']
    for r in nodes.itertuples():
        y = ROW * (r.node_id + 1)
        colour = COLOURS.get(r.type, CLAUSE_COLOUR)
        label = r.type + (" ⟲" if r.backtracking_marker else "") + (" ✓" if r.verification_marker else "")
        snippet = html.escape(str(r.text if isinstance(r.text, str) else "")[:45].replace("\n", " "))
        svg.append(f'<g class="gnode" data-id="{r.node_id}"><circle cx="{spine}" cy="{y}" r="7" fill="{colour}"/>'
                   f'<text x="{spine + 12}" y="{y + 4}" font-size="11">{r.node_id} {html.escape(label)}</text>'
                   f'<text x="{spine + 125}" y="{y + 4}" font-size="9" fill="#777">{snippet}</text></g>')
    for e in edge_rows.itertuples():
        y1, y2 = ROW * (e.source_node_id + 1), ROW * (e.target_node_id + 1)
        colour, dash = EDGE_STYLE[e.edge_source]
        bulge = min(20 + 6 * (e.target_node_id - e.source_node_id), spine - 12)
        if e.edge_source != "discourse":
            bulge = min(bulge + 30, spine - 12)   # tool/similarity/identifier arcs sit further out so they don't overlap parser arcs
        x = spine - bulge
        path = f"M {spine - 7} {y1} C {x} {y1}, {x} {y2}, {spine - 7} {y2}"
        label = html.escape(e.relation if isinstance(e.relation, str) else e.edge_source[:3]) + (f"/{e.nuclearity}" if isinstance(e.nuclearity, str) else "")
        svg.append(f'<g class="gedge" data-s="{e.source_node_id}" data-t="{e.target_node_id}">'
                   f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.2" stroke-dasharray="{dash}"/>'
                   f'<text x="{x + (spine - 7 - x) / 4}" y="{(y1 + y2) / 2 + 3}" font-size="8" fill="{colour}" text-anchor="middle">{label}</text></g>')
    svg.append("</svg>")
    return (f'<section class="case"><h2>{html.escape(title)}</h2><p class="summary">{html.escape(summary)}</p>'
            f'<div class="cols"><div class="text"><pre>{text_html}</pre></div><div class="graph">{"".join(svg)}</div></div>'
            f'{parameters}</section>')


STYLE = """
body{font:13px/1.4 -apple-system,Helvetica,sans-serif;margin:0;padding:1em 1.5em}
nav{position:sticky;top:0;background:#fff;padding:.4em 0;border-bottom:1px solid #ddd;font-size:12px}
.case{border-top:2px solid #999;margin-top:2em;padding-top:.5em}
.cols{display:grid;grid-template-columns:1fr 640px;gap:1.5em;align-items:start}
.text pre{white-space:pre-wrap;word-wrap:break-word;font:12px/1.5 Menlo,monospace;max-height:85vh;overflow:auto}
.graph{overflow:auto;max-height:85vh}
.node{border-left:3px solid;padding-left:3px;margin-left:1px}
.node b{color:#fff;font-size:9px;padding:0 3px;border-radius:2px;margin-right:3px}
.node.hl{background:#ffe9a8}.gnode.hl circle{stroke:#000;stroke-width:3}.gedge.hl path{stroke-width:3}.gedge.hl text{font-weight:bold}
.summary{margin:.2em 0 .6em;font-size:12px;color:#555}
.params{margin-top:.8em;border-top:1px solid #ddd;padding-top:.4em}
.params summary{cursor:pointer;color:#1f4e9c;font-size:12px}
.params h3{font-size:12px;margin:.9em 0 .3em;color:#333}
.rows{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:0 18px}
.frow{display:flex;align-items:center;gap:6px;font-size:11px;padding:1px 0;border-bottom:1px solid #f0f0f0}
.frow:hover{background:#f6f8fc}
.fname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fval{width:62px;text-align:right;font:10px Menlo,monospace;color:#111}
.fbar{width:54px;height:8px;flex:none;border-radius:2px;overflow:hidden;
 background:linear-gradient(to right,#eee 0 50%,#ccc 50% calc(50% + 1px),#eee calc(50% + 1px) 100%)}
.fbar i{display:block;height:100%;background:#1f4e9c}
.fpct{width:24px;text-align:right;color:#888;font-size:10px}
"""
SCRIPT = """
document.querySelectorAll('.case').forEach(c=>{
 const set=(id,on)=>{c.querySelectorAll(`.node[data-id="${id}"],.gnode[data-id="${id}"]`).forEach(x=>x.classList.toggle('hl',on));
  c.querySelectorAll(`.gedge[data-s="${id}"],.gedge[data-t="${id}"]`).forEach(x=>x.classList.toggle('hl',on));};
 c.querySelectorAll('[data-id]').forEach(el=>{el.addEventListener('mouseenter',()=>set(el.dataset.id,true));el.addEventListener('mouseleave',()=>set(el.dataset.id,false));});
});
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-bench", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--with-plans", action="store_true", help="only transcripts that state a plan (agentic benchmarks)")
    args = parser.parse_args()
    nodes = pd.read_parquet(NETWORK / "nodes.parquet")
    edges = pd.read_parquet(NETWORK / "edges.parquet")
    plan_path = NETWORK / "plan_edges.parquet"
    if plan_path.exists():                       # plan item -> the tool call that carries it out (plan_edges.py)
        plan = pd.read_parquet(plan_path)
        plan = plan.assign(edge_source="plan", relation=plan.overlap.map({1.0: "realises", 0.0: "nearest"}), nuclearity=None)
        edges = pd.concat([edges, plan[["transcript_key", "source_node_id", "target_node_id", "edge_source", "relation", "nuclearity", "cosine"]]], ignore_index=True)
    nodes = nodes.sort_values(["transcript_key", "node_id"]).reset_index(drop=True)
    nodes["type"] = node_types(nodes, edges)
    store = pd.read_parquet(FREEZE / "transcript_store.parquet", columns=["content_hash", "transcript"]).drop_duplicates("content_hash").set_index("content_hash")
    panels = feature_panels(nodes.drop_duplicates("transcript_key")[KEYS + ["transcript_key"]])
    sizes = nodes.groupby("transcript_key").size()
    sections, titles = [], []
    for bench in ["gpqa", "aime2024", "aime2025", "intercode", "swe_bench_mini"]:
        keys = nodes[(nodes.bench == bench) & (nodes.transcript_key.map(sizes).between(12, args.max_nodes))].transcript_key.drop_duplicates()
        if args.with_plans:
            keys = keys[keys.isin(set(edges.transcript_key[edges.edge_source == "plan"]))]
        for key in keys.sample(min(args.per_bench, len(keys)), random_state=args.seed):
            group = nodes[nodes.transcript_key == key]
            head = group.iloc[0]
            own_edges = edges[edges.transcript_key == key]
            title = f"{head['bench']} | {head['model']} | item {head['item']}"
            titles.append(title)
            sections.append(render(store.transcript[head["content_hash"]], group, own_edges, f"{len(sections) + 1}. {title}",
                                   summary_line(group, own_edges, panels, key), parameters_html(panels, key)))
    legend = ("<p>Left: transcript, each node's span numbered and coloured by type (clauses dark grey, equations red, headings purple, "
              "answer green, tool calls blue, tool outputs light blue / orange on error). Right: nodes in reading order; arcs = edges: "
              "discourse parser (black, relation/nuclearity), tool flow (blue), similarity (grey dashed), shared identifier (green), plan item → the call that realises it (gold dashed; 'nearest' = no class match); longer arcs bulge further out. "
              "⟲ backtracking marker, ✓ verification marker. Hover a node on either side. Under each pair, "
              "'graph parameters' opens that transcript's topology, graph descriptors, shapes and motifs, plan and "
              "exposition features, each with the percentile it sits at among its benchmark's transcripts.</p>")
    nav = "<nav>" + " | ".join(f'<a href="#case-{i + 1}">{i + 1} {html.escape(t.split(" | item")[0])}</a>' for i, t in enumerate(titles)) + "</nav>"
    sections = [sec.replace('<section class="case">', f'<section class="case" id="case-{i + 1}">', 1) for i, sec in enumerate(sections)]
    global OUT
    if args.with_plans:
        OUT = OUT.with_name("graph_viewer_plans.html")
    OUT.write_text(f"<!doctype html><meta charset=utf-8><title>Clause graphs</title><style>{STYLE}</style>"
                   f"<h1>Clause network: transcript beside graph</h1>{legend}{nav}{''.join(sections)}<script>{SCRIPT}</script>")
    print(f"wrote {OUT} with {len(sections)} transcripts")


if __name__ == "__main__":
    main()
