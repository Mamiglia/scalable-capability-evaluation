import numpy as np
import pandas as pd

from plan_edges import plan_items, match_plan, plan_features


def _group():
    rows = [("question", None, "find the flag"),
            ("assistant_edu", 0.0, "Plan:"), ("assistant_edu", 0.0, "- List the current directory"),
            ("assistant_edu", 0.0, "to find the binary."), ("assistant_edu", 0.0, "- Inspect the binary with file and strings"),
            ("assistant_edu", 0.0, "- Submit the flag."),
            ("tool_call", 0.0, "[Tool Call: bash({'command': 'ls -la'})]"), ("tool_output", 1.0, "files"),
            ("tool_call", 2.0, "[Tool Call: bash({'command': 'file static'})]"), ("tool_output", 3.0, "ELF"),
            ("tool_call", 4.0, "[Tool Call: submit({'answer': 'picoCTF{x}'})]")]
    return pd.DataFrame({"transcript_key": "k", "node_id": range(len(rows)), "kind": [r[0] for r in rows],
                         "part_id": [r[1] for r in rows], "text": [r[2] for r in rows]})


def test_plan_items_group_marker_clauses_with_their_continuations():
    items = plan_items(_group())
    assert [i[0] for i in items] == [2, 4, 5]
    assert items[0][1] == "List the current directory to find the binary."


def test_match_by_command_class_in_plan_order():
    group = _group()
    embeddings = np.zeros((len(group), 4), dtype=np.float32)
    embeddings[:, 0] = 1.0                               # every node identical: embeddings cannot decide
    edges = match_plan(group, embeddings, plan_items(group))
    assert [e["target_node_id"] for e in edges] == [6, 8, 10]      # list -> ls, read -> file static, submit -> submit
    assert all(e["overlap"] == 1.0 for e in edges)
    f = plan_features(group, edges)
    assert f["plan_items"] == 3 and f["plan_executed_share"] == 1.0 and f["plan_order_concordance"] == 1.0 and f["plan_unplanned_share"] == 0.0
