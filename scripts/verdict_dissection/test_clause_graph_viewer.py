import pandas as pd

from clause_graph_viewer import show, summary_line, within_bench_percentiles


def test_percentiles_pool_the_aime_years_and_rank_within_benchmark():
    frame = pd.DataFrame({"bench": ["aime2024", "aime2025", "aime2026", "gpqa", "gpqa"],
                          "density": [1.0, 2.0, 3.0, 9.0, 1.0]})
    got = within_bench_percentiles(frame, ["density"]).density.tolist()
    assert got == [1 / 3, 2 / 3, 1.0, 1.0, 0.5]      # the AIME years share one ranking, GPQA has its own


def test_show_keeps_counts_whole_and_shortens_fractions():
    assert show(30.0) == "30" and show(0.7512345) == "0.751" and show(float("nan")) == "–"


def test_summary_counts_edges_by_source_and_reports_the_plan():
    group = pd.DataFrame({"node_id": range(4)})
    edges = pd.DataFrame({"edge_source": ["discourse", "discourse", "tool", "plan"]})
    panels = [("Topology", ["topo_aug_longest_path"], pd.DataFrame({"topo_aug_longest_path": [5.0]}, index=["k"]), None),
              ("Plan", ["plan_items", "plan_executed_share"],
               pd.DataFrame({"plan_items": [2.0], "plan_executed_share": [0.5]}, index=["k"]), None)]
    assert summary_line(group, edges, panels, "k") == (
        "4 nodes · 4 edges (2 discourse, 1 tool, 1 plan) · longest path 5 · 2 plan items, 50% executed")
