import pandas as pd

from features_transitions import node_types, transition_shares


def test_node_types_use_nearest_discourse_parent_and_tool_fields():
    nodes = pd.DataFrame({"transcript_key": ["k"] * 5, "node_id": range(5),
                          "kind": ["question", "assistant_edu", "assistant_edu", "equation", "tool_output"],
                          "tool_type": [None] * 5, "outcome": [None, None, None, None, "error"]})
    edges = pd.DataFrame({"transcript_key": ["k", "k", "k"], "source_node_id": [1, 2, 1], "target_node_id": [2, 3, 2],
                          "edge_source": ["discourse", "similarity", "discourse"], "relation": ["causal", None, "elaboration"]})
    assert list(node_types(nodes, edges)) == ["question", "causal", "elaboration", "equation", "output_error"]   # node 1 has no parent: typed by its first rhetorical edge as source


def test_transition_shares_count_one_and_two_step_pairs():
    row = transition_shares(["question", "root", "equation", "elaboration", "equation"])
    assert row["n_nodes"] == 4
    assert row["t1__equation__elaboration"] == 1 / 3 and row["t1__elaboration__equation"] == 1 / 3
    assert row["t2__equation__equation"] == 1 / 2 and row["t2__root__elaboration"] == 1 / 2
