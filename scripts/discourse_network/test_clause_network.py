"""Small checks for the clause-network pipeline, one synthetic transcript per module."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.nodes import (build_nodes, leaf_to_clauses, mask_prose, output_outcome,  # noqa: E402
                                     protected_spans, tool_type)

LEXICON = {"inspect": ["ls", "cat", "grep", "sed -n", "echo"], "act": ["sed -i", "rm", "submit", ">"]}
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


def test_mask_prose_hides_markdown_emphasis_without_moving_offsets():
    text = "Use **this title:** now."
    masked, cmap, _ = mask_prose(text)
    assert masked == "Use   this title:   now."
    assert len(masked) == len(text) and cmap == list(range(len(text) + 1))


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
    assert tool_type("bash", "[Tool Call: bash({'command': 'cat f > out'})]", LEXICON, WRITE) == "act"
    assert tool_type("bash", "[Tool Call: bash({'command': 'echo format'})]", LEXICON, WRITE) == "inspect"
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


def test_nodes_use_the_last_nonempty_assistant_part_as_the_sink():
    text = "Done."
    parts = [assistant_part(text),
             {"role": "assistant", "content": "", "start": 30, "end": 30, "bare": False, "subparts": []}]
    nodes = build_nodes("Q?", "[Assistant]: " + text, parts)
    assert nodes[-1]["is_final_sink"]


def test_snap_to_words_and_merge_fragments():
    from discourse_network.nodes import merge_fragments, snap_to_words
    text = "We get 7.\n\nTotal for this pattern: 147. Now, however, the count is 6."
    raw = [(0, 12), (12, 34), (35, 39), (40, 48), (49, 64)]        # parser cut one char into "Total"; "Now," and "however," fragments
    snapped = snap_to_words(raw, text)
    assert [text[a:b] for a, b in snapped] == ["We get 7.", "Total for this pattern:", "147.", "Now,", "however, the count"]
    merged = merge_fragments(snapped, text)
    assert [text[a:b] for a, b in merged] == ["We get 7.", "Total for this pattern:", "147.", "Now, however, the count"]


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


def test_segment_text_repairs_parser_gaps_at_word_boundaries():
    text = "Every resident owns H, so the ownership continues."
    so, following = text.index("so"), text.index("the")

    class GapParser:
        def __call__(self, _):
            return {"rst": [Tree(Leaf(0, so + 1), Leaf(following, len(text)), "elaboration", "NS")]}

    clauses, trees = segment_text(GapParser(), text)
    assert [text[a:b] for a, b in clauses] == [
        "Every resident owns H, so", "the ownership continues."]
    assert trees[0]["left"] == {"start": 0, "end": so + 2}
    assert trees[0]["right"] == {"start": following, "end": len(text)}


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


def test_discourse_edges_collapse_leaves_merged_into_one_node():
    text = "Now, continue."
    nodes = [node(0, "question", None), node(1, "assistant_edu", 0, 13, 27, text)]
    tree = {"start": 0, "end": len(text), "relation": "joint", "nuclearity": "NN",
            "left": {"start": 0, "end": 4}, "right": {"start": 5, "end": len(text)}}
    assert discourse_edges(nodes, {(0, 0): [tree]}, {(0, 0): (text, 13)}) == []


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


# ---------------------------------------------------------------- features

from discourse_network.features import EdgeSpace, TopKSAE, sae_block, topology_block  # noqa: E402


def e(s, t, source, relation=None):
    return {"source_node_id": s, "target_node_id": t, "edge_source": source, "relation": relation,
            "nuclearity": None, "cosine": None}


def test_typed_topology_shares_and_motifs():
    nodes = [node(0, "question", None), node(1, "assistant_edu", 0),
             node(2, "tool_call", 0, call_id="a", tool_type="inspect"),
             node(3, "tool_output", 1, call_id="a", tool_type="inspect", outcome="ok"),
             node(4, "tool_call", 2, call_id="b", tool_type="act"),
             node(5, "tool_output", 3, call_id="b", tool_type="act", outcome="error"),
             node(6, "assistant_edu", 4, backtracking_marker=True), node(7, "assistant_edu", 4, verification_marker=True),
             node(8, "answer", 4, is_final_sink=True)]
    edges = [e(0, 1, "similarity"), e(1, 2, "discourse", "purpose"), e(2, 3, "tool", "tool_result"),
             e(3, 4, "tool", "tool_response"), e(4, 5, "tool", "tool_result"), e(5, 6, "tool", "tool_response"),
             e(6, 7, "discourse", "causal"), e(7, 8, "discourse", "restatement")]
    f = topology_block(nodes, edges)
    assert f["topo_node_count"] == 9 and f["topo_edge_count"] == 8 and f["topo_longest_path"] == 8
    assert f["topo_share_tool_inspect"] == 1 / 9 and f["topo_share_tool_act"] == 1 / 9
    assert f["topo_output_error_share"] == 0.5
    assert f["topo_relation_causal"] == 1 / 8 and f["topo_edge_source_tool"] == 0.5
    assert f["topo_error_then_correction"] == 1.0 and f["topo_act_after_inspect"] == 1.0
    assert f["topo_final_ancestor_share"] == 1.0 and f["topo_ancestry_share_tool_act"] == 1 / 9
    assert f["topo_restatement_chain_max"] == 1.0 and f["topo_weak_components"] == 1.0


def test_motifs_only_use_the_next_assistant_part():
    nodes = [node(0, "question", None),
             node(1, "tool_output", 1, outcome="error"),
             node(2, "assistant_edu", 2, backtracking_marker=False),
             node(3, "assistant_edu", 4, backtracking_marker=True, is_final_sink=True)]
    assert topology_block(nodes, [])["topo_error_then_correction"] == 0.0


def test_sae_is_sparse_and_aggregates_per_transcript():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 16))
    sae = TopKSAE.fit(x, width=8, top_k=2, iterations=2, batch_size=64)
    z = sae.transform(x)
    assert z.shape == (400, 8) and (z > 0).sum(axis=1).max() <= 2
    block = sae_block(z, np.repeat(np.arange(4), 100), n_transcripts=4, prefix="node_sae", min_support=10)
    assert len(block) == 4 and block.filter(like="_active").max().max() <= 1.0
    assert any(c.endswith("_mean") for c in block.columns)


def test_sae_support_is_selected_on_development_units_only():
    z = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    block = sae_block(z, [0, 0, 1, 1], 2, "node_sae", min_support=2,
                      support_mask=np.array([True, True, False, False]))
    assert block.columns.tolist() == ["node_sae_f00_mean", "node_sae_f00_active"]


def test_edge_space_is_pca_of_pair_features():
    emb = np.eye(5, dtype=np.float32)
    edges = [{"source_node_id": i, "target_node_id": i + 1} for i in range(4)] * 5
    space = EdgeSpace.fit(emb, edges, components=3)
    assert space.transform(emb, edges).shape == (20, 3)


# ---------------------------------------------------------------- build (end to end with fakes)

import pandas as pd  # noqa: E402

from discourse_network import build  # noqa: E402


def fake_store(n=6):
    rows = []
    for i in range(n):
        text = f"Let x{i}=1, so y={i}.\n\\[\nz={i}\n\\]\nANSWER: {i}"
        rows.append({"bench": "gpqa" if i < 4 else "intercode", "model": f"m{i % 2}", "item": f"it{i // 2}", "epoch": 0,
                     "log": "l", "sample_epoch": 1, "content_hash": f"h{i}", "question": "Q?",
                     "transcript": "[Assistant]: " + text, "transcript_parts_json": json.dumps([assistant_part(text)]),
                     "correct": True, "is_canonical": True})
    return pd.DataFrame(rows)


def fake_segments(store):
    """What segment.py would write: two clauses per transcript and one causal tree over the masked text."""
    clauses, trees = [], []
    for r in store.itertuples():
        text = json.loads(r.transcript_parts_json)[0]["content"]
        cut, prose_end = text.index(", ") + 1, text.index("\n\\[")
        keys = {k: getattr(r, k) for k in build.KEYS} | {"part_id": 0, "subpart_id": 0}
        clauses += [keys | {"start": 13, "end": 13 + cut}, keys | {"start": 13 + cut + 1, "end": 13 + prose_end}]
        masked_len = len(text[:prose_end]) + len("\nEQUATION\nEQUATION")
        trees.append(keys | {"tree_json": json.dumps(
            {"start": 0, "end": masked_len, "relation": "causal", "nuclearity": "NS",
             "left": {"start": 0, "end": cut}, "right": {"start": cut + 1, "end": masked_len}})})
    return pd.DataFrame(clauses), pd.DataFrame(trees)


def fake_embed(texts, cfg):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(len(texts), 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_pipeline_writes_tables(tmp_path):
    store = fake_store()
    clauses, trees = fake_segments(store)
    split = pd.DataFrame({"bench": store.bench, "item": store.item, "split": "representation_train"})
    verdicts = pd.DataFrame([{"bench": "gpqa", "judge": "j", "item": "it1", "model_left": "m1", "model_right": "m0",
                              "epoch_left": 0, "epoch_right": 0, "log_left": "l", "log_right": "l", "pick_left": 1}])
    cfg = {"edges": {"max_parents": 3, "k_similar": 1, "similarity_threshold": -1.0},
           "sae": {"width": 4, "top_k": 2, "iterations": 1, "batch_size": 8, "seed": 0, "min_support": 1,
                   "edge_pca_components": 2, "benchmark_groups": {"gpqa": ["gpqa"], "intercode": ["intercode"]}},
           "encoder": {}, "tool_types": {"inspect": ["ls"], "act": ["rm"]}, "python_write_markers": [],
           "error_markers": ["Traceback"]}
    build.run(store, clauses, trees, split, verdicts, cfg, tmp_path, embed=fake_embed)
    nodes = pd.read_parquet(tmp_path / "nodes.parquet")
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    feats = pd.read_parquet(tmp_path / "transcript_features.parquet")
    pairs = pd.read_parquet(tmp_path / "pair_features.parquet")
    assert nodes.kind.value_counts()["equation"] == 6 and nodes.kind.value_counts()["answer"] == 6
    assert set(edges.edge_source) >= {"discourse", "similarity"} and (edges.relation == "causal").sum() == 6
    assert len(feats) == 6 and feats.topo_relation_causal.gt(0).all()
    assert any(c.startswith("node_sae_") for c in feats) and any(c.startswith("edge_sae_") for c in feats)
    for group in ("gpqa", "intercode"):
        assert any(c.startswith(f"node_sae_{group}_") for c in feats)
        assert any(c.startswith(f"edge_sae_{group}_") for c in feats)
    assert feats.loc[feats.bench == "intercode"].filter(like="node_sae_gpqa_").isna().all().all()
    assert feats.loc[feats.bench == "gpqa"].filter(like="_sae_").notna().any().any()
    assert len(pairs) == 1 and "d_topo_node_count" in pairs
    assert json.loads((tmp_path / "manifest.json").read_text())["rows"]["transcripts"] == 6
    assert not list(tmp_path.rglob("*.partial"))


def test_pipeline_can_stop_after_nodes(tmp_path):
    store = fake_store()
    clauses, trees = fake_segments(store)
    split = pd.DataFrame({"bench": store.bench, "item": store.item, "split": "representation_train"})
    cfg = {"edges": {}, "sae": {}, "encoder": {}, "tool_types": {},
           "python_write_markers": [], "error_markers": []}
    result = build.run(store, clauses, trees, split, pd.DataFrame(), cfg, tmp_path,
                       embed=lambda *_: (_ for _ in ()).throw(AssertionError("embedded")), stop_after="nodes")
    assert len(result) and (tmp_path / "nodes.parquet").exists()
    assert not (tmp_path / "node_embeddings.npy").exists()


def test_build_import_does_not_require_the_pair_builder():
    code = ("import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/verdict_dissection'); "
            "sys.modules['aggregate_surprisal'] = None; import discourse_network.build")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_atomic_write_refuses_an_existing_partial(tmp_path):
    path = tmp_path / "output.parquet"
    partial = tmp_path / "output.parquet.partial"
    partial.write_text("incomplete")
    with pytest.raises(FileExistsError):
        build._atomic(path, lambda target: target.write_text("replacement"))
    assert partial.read_text() == "incomplete" and not path.exists()


def test_make_nodes_keeps_rows_with_the_same_content_hash_separate():
    store = fake_store(2)
    for column in ("content_hash", "question", "transcript", "transcript_parts_json"):
        store.loc[1, column] = store.loc[0, column]
    clauses, _ = fake_segments(store)
    cfg = {"tool_types": {}, "python_write_markers": [], "error_markers": []}
    nodes = build.make_nodes(store, clauses, cfg)
    assert nodes.groupby("transcript_key").size().tolist() == [5, 5]
