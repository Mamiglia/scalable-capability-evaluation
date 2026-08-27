from features_graph_descriptors import descriptors


def test_chain_versus_star():
    chain = descriptors(["a"] * 5, [(0, 1), (1, 2), (2, 3), (3, 4)], ["joint"] * 4)
    star = descriptors(["a"] * 5, [(0, 1), (0, 2), (0, 3), (0, 4)], ["joint"] * 4)
    assert chain["diameter"] == 4 and star["diameter"] == 2
    assert chain["adjacent_edge_share"] == 1.0 and star["adjacent_edge_share"] == 0.25
    assert star["betweenness_centralisation"] > chain["betweenness_centralisation"]
    assert chain["cyclomatic"] == 0 and chain["density"] == 0.4
    assert chain["source_share"] == 0.2 and chain["sink_share"] == 0.2
