"""Small checks for the clause-network pipeline, one synthetic transcript per module."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.nodes import (build_nodes, leaf_to_clauses, mask_prose, output_outcome,  # noqa: E402
                                     protected_spans, tool_type)

LEXICON = {"inspect": ["ls", "cat", "grep", "sed -n"], "act": ["sed -i", "rm", "submit"]}
WRITE = ["'w'", "os.remove"]
ERRORS = ["Traceback", "command not found"]


def assistant_part(text, start=13, extra_subparts=()):
    """One assistant part whose content subpart is `text`, rendered after '[Assistant]: '."""
    sub = [{"kind": "content", "content": text, "start": start, "end": start + len(text),
            "call_id": None, "function": None, "raw_payload": None}] + list(extra_subparts)
    return {"role": "assistant", "content": text, "start": start, "end": sub[-1]["end"], "bare": False,
            "subparts": sub}


# ---------------------------------------------------------------- nodes

def test_protected_spans_cover_headings_and_answer():
    text = "### Plan\nWe have\n$$x=1$$\nANSWER: 3"
    assert [(text[s:e], k) for s, e, k in protected_spans(text)] == [
        ("### Plan", "heading"), ("$$x=1$$", "equation"), ("ANSWER: 3", "answer")]


def test_mask_prose_replaces_display_and_inline_maths_and_maps_back():
    text = "The sum is \\(99\\%\\) of this:\n\\[\nm=1\n\\]\nThus x."
    masked, cmap, protected = mask_prose(text)
    assert masked == "The sum is MATH of this:\nEQUATION\nThus x."
    assert protected == [(29, 38, "equation")]
    assert text[cmap[masked.index("Thus")]:].startswith("Thus")
    assert cmap[len(masked)] == len(text)
    assert mask_prose("Check `use_required_attribute` next.")[0] == "Check MATH next."


def test_leaf_to_clauses_drops_protected_spans_inside_a_leaf():
    text = "Thus\n\n\\[\na=1\n\\]\n\\[\nb=2\n\\]"
    masked, cmap, protected = mask_prose(text)
    assert leaf_to_clauses(0, len(masked), cmap, text, protected) == [(0, 4)]


def test_tool_type_and_outcome():
    assert tool_type("bash", "[Tool Call: bash({'command': 'ls -la'})]", LEXICON, WRITE) == "inspect"
    assert tool_type("bash", "[Tool Call: bash({'command': 'sed -i s/a/b/ f.py'})]", LEXICON, WRITE) == "act"
    assert tool_type("python", "[Tool Call: python({'code': 'print(open(\"f\").read())'})]", LEXICON, WRITE) == "inspect"
    assert tool_type("python", "[Tool Call: python({'code': \"open('f','w').write(1)\"})]", LEXICON, WRITE) == "act"
    assert tool_type("submit", "[Tool Call: submit({'answer': 'x'})]", LEXICON, WRITE) == "act"
    assert output_outcome("[Tool Output (bash)]: bash: xxd: command not found", ERRORS) == "error"
    assert output_outcome("[Tool Output (bash)]: total 12", ERRORS) == "ok"


def test_nodes_are_in_character_order_with_segmenter_clauses():
    text = "Let c=1,\nso masses are energies.\n\\[\nM=300\n\\]\nThus\nANSWER: A"
    transcript = "[Assistant]: " + text
    span = lambda s: (transcript.index(s), transcript.index(s) + len(s))
    clauses = {(0, 0): [span("Let c=1,"), span("so masses are energies."), span("Thus")]}
    nodes = build_nodes("Q?", transcript, [assistant_part(text)], clauses)
    assert [n["kind"] for n in nodes] == ["question", "assistant_edu", "assistant_edu", "equation",
                                          "assistant_edu", "answer"]
    assert [n["text"] for n in nodes[1:3]] == ["Let c=1,", "so masses are energies."]
    assert [n["start"] for n in nodes[1:]] == sorted(n["start"] for n in nodes[1:])
    assert nodes[-1]["is_final_sink"] and nodes[-1]["text"] == "ANSWER: A"


def test_nodes_fall_back_to_sentences_and_type_tools():
    call = "[Tool Call: bash({'command': 'cat f'})]"
    out = "[Tool Output (bash)]: Traceback (most recent call last)"
    transcript = "[Assistant]: Look. Then look again.\n" + call + "\n\n[Tool]: " + out
    call_start = transcript.index(call)
    out_start = transcript.index(out)
    parts = [assistant_part("Look. Then look again.", extra_subparts=[
                 {"kind": "tool_call", "content": call, "start": call_start, "end": call_start + len(call),
                  "call_id": "c1", "function": "bash", "raw_payload": "{'command': 'cat f'}"}]),
             {"role": "tool", "content": out, "start": out_start, "end": out_start + len(out), "bare": False,
              "subparts": [{"kind": "tool_output", "content": out, "start": out_start, "end": out_start + len(out),
                            "call_id": "c1", "function": "bash", "raw_payload": None}]}]
    nodes = build_nodes("Q?", transcript, parts, tool_lexicon=LEXICON, python_write_markers=WRITE,
                        error_markers=ERRORS)
    assert [n["kind"] for n in nodes] == ["question", "assistant_edu", "assistant_edu", "tool_call", "tool_output"]
    assert nodes[3]["tool_type"] == "inspect" and nodes[3]["is_final_sink"]
    assert nodes[4]["tool_type"] == "inspect" and nodes[4]["outcome"] == "error"


# ---------------------------------------------------------------- segment

from discourse_network.segment import segment_rows, segment_text  # noqa: E402


class Leaf:
    def __init__(self, start, end):
        self.left = self.right = None
        self.start, self.end, self.relation, self.nuclearity = start, end, None, None


class Tree:
    def __init__(self, left, right, relation, nuclearity):
        self.left, self.right, self.relation, self.nuclearity = left, right, relation, nuclearity
        self.start, self.end = left.start, right.end


class FakeParser:
    """Splits the masked text once at ', ' and joins the halves with elaboration/NS."""
    def __call__(self, text):
        cut = text.index(", ") + 1
        return {"rst": [Tree(Leaf(0, cut), Leaf(cut + 1, len(text)), "elaboration", "NS")]}


def test_segment_text_maps_clauses_and_tree_back_to_the_original():
    text = "Let \\(c=1\\), so\n\\[\nM=1\n\\]\nholds."
    clauses, trees = segment_text(FakeParser(), text)
    assert [text[a:b] for a, b in clauses] == ["Let \\(c=1\\),", "so", "holds."]
    assert trees[0]["relation"] == "elaboration" and trees[0]["left"] == {"start": 0, "end": 9}


def test_segment_rows_uses_transcript_offsets():
    text = "Let x, go."
    row = {"bench": "b", "model": "m", "item": "i", "epoch": 0, "log": "l", "sample_epoch": 1,
           "content_hash": "h", "transcript_parts_json": json.dumps([assistant_part(text)])}
    clauses, trees = segment_rows([row], FakeParser())
    assert clauses[["start", "end"]].values.tolist() == [[13, 19], [20, 23]]
    assert json.loads(trees.tree_json.iloc[0])["nuclearity"] == "NS"


# ---------------------------------------------------------------- edges

from discourse_network.edges import cap_parents, discourse_edges, long_range_edges, tool_edges  # noqa: E402


def node(i, kind, part, start=None, end=None, text="", **extra):
    base = {"node_id": i, "kind": kind, "part_id": part, "start": start, "end": end, "text": text,
            "role": "tool" if kind == "tool_output" else ("question" if kind == "question" else "assistant"),
            "call_id": None, "tool_type": None, "outcome": None, "is_final_sink": False}
    return base | extra


def test_discourse_edges_follow_nucleus_heads():
    text = "A, B. \\[\nx\n\\]"                    # masked: "A, B. EQUATION"
    nodes = [node(0, "question", None), node(1, "assistant_edu", 0, 13, 15, "A,"),
             node(2, "assistant_edu", 0, 16, 18, "B."), node(3, "equation", 0, 19, 13 + len(text))]
    tree = {"start": 0, "end": 14, "relation": "causal", "nuclearity": "SN",
            "left": {"start": 0, "end": 2},
            "right": {"start": 3, "end": 14, "relation": "restatement", "nuclearity": "NN",
                      "left": {"start": 3, "end": 5}, "right": {"start": 6, "end": 14}}}
    edges = discourse_edges(nodes, {(0, 0): [tree]}, {(0, 0): (text, 13)})
    assert {(e["source_node_id"], e["target_node_id"]): e["relation"] for e in edges} == {
        (1, 2): "causal", (2, 3): "restatement"}


def test_tool_edges_link_outputs_to_calls_and_to_the_next_assistant_part():
    nodes = [node(0, "question", None), node(1, "assistant_edu", 0, 13, 20),
             node(2, "tool_call", 0, 21, 40, call_id="c1"), node(3, "tool_output", 1, 50, 60, call_id="c1"),
             node(4, "assistant_edu", 2, 70, 80)]
    assert [(e["source_node_id"], e["target_node_id"], e["relation"]) for e in tool_edges(nodes)] == [
        (2, 3, "tool_result"), (3, 4, "tool_response")]


def test_long_range_edges_use_cosine_and_identifiers_outside_own_part():
    nodes = [node(0, "question", None, text="find flag_1"), node(1, "assistant_edu", 0, 13, 20, "look for flag_1"),
             node(2, "assistant_edu", 0, 21, 30, "then decode"), node(3, "assistant_edu", 2, 40, 50, "decode it now")]
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 1.0]])
    edges = long_range_edges(nodes, emb, k_similar=1, threshold=0.6)
    assert {(e["source_node_id"], e["target_node_id"], e["edge_source"]) for e in edges} == {
        (0, 1, "identifier"), (0, 1, "similarity"), (2, 3, "similarity")}


def test_cap_parents_keeps_structure_first_then_best_cosine():
    edges = [{"source_node_id": 1, "target_node_id": 5, "edge_source": "similarity", "cosine": 0.9},
             {"source_node_id": 2, "target_node_id": 5, "edge_source": "discourse", "cosine": None},
             {"source_node_id": 3, "target_node_id": 5, "edge_source": "similarity", "cosine": 0.7},
             {"source_node_id": 4, "target_node_id": 5, "edge_source": "tool", "cosine": None}]
    assert sorted(e["source_node_id"] for e in cap_parents(edges, max_parents=3)) == [1, 2, 4]
