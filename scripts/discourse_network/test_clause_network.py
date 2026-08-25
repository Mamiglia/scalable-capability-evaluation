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
