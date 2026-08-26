import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_annotators import aligned_labels, kappa, merge_tables

PROMPT = 'CONCEPTS:\n1. "long"\n2. "short"\n\nTRANSCRIPT 1:\nx\n\nTRANSCRIPT 2:\ny\n\nTRANSCRIPT 3:\nz\n'


def test_labels_align_by_prompt_and_kappa_is_zero_for_chance_agreement(tmp_path):
    for name, key, reply in (("a", "ka", '{"1": [1, 2], "2": [3]}'), ("b", "kb", '{"1": [1], "2": [3]}')):
        (tmp_path / f"pending_{name}.jsonl").write_text(json.dumps({"key": key, "model": name, "prompt": PROMPT}) + "\n")
        (tmp_path / f"cache_{name}.jsonl").write_text(json.dumps({"key": key, "model": name, "reply": reply}) + "\n")
    a, b = aligned_labels(tmp_path / "pending_a.jsonl", tmp_path / "cache_a.jsonl",
                          tmp_path / "pending_b.jsonl", tmp_path / "cache_b.jsonl")
    assert a.tolist() == [1, 1, 0, 0, 0, 1] and b.tolist() == [1, 0, 0, 0, 0, 1]
    assert kappa(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == 0.0
    assert kappa(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0])) == 1.0


def test_merge_marks_cells_kept_by_both():
    a = pd.DataFrame({"feature": ["f@g", "h@g"], "group": ["g", "g"], "prevalence": [0.5, 0.5],
                      "description": ["x", "y"], "correlation": [0.4, 0.1], "p_value": [0.001, 0.5], "kept": [True, False]})
    b = a.assign(description=["x", "z"], kept=[True, True])
    out = merge_tables(a, b, "codex", "haiku")
    assert out.kept_both.tolist() == [True, False] and out.same_description.tolist() == [True, False]
