from features_graph_motifs import graph_features


def test_options_converging_and_plan_fan_out():
    # plan node 0 fans out to three option clauses 1,2,3 which all converge on the answer 4
    types = ["organization", "joint", "joint", "joint", "answer"]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
    f, wl = graph_features(types, edges, [None] * 5)
    assert f["plan_fan_out"] == 3 and f["plan_coverage"] == 4 / 5
    assert f["answer_in_degree"] == 3 and f["sibling_convergence_max"] == 3
    assert f["motif_fan_out"] == 3 / 5 and f["motif_fan_in"] == 3 / 5 and f["motif_chain"] == 3 / 5
    assert wl["wl__answer|joint,joint,joint"] == 1 / 5


def test_tool_then_change_of_mind():
    types = ["tool_inspect", "output_ok", "adversative", "tool_act", "output_error", "causal"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    f, _ = graph_features(types, edges, [None, None, "backtracking", None, None, "backtracking"])
    assert f["after_tool_backtrack"] == 1.0 and f["after_tool_adversative"] == 0.5
    assert f["after_error_backtrack"] == 1.0 and f["sequential_share"] == 1.0
