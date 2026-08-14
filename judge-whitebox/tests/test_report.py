import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest
import torch

import importance_report as ir
import wbconfig


def _screen_result(activity, attr_signed, attr_abs, n_tokens, n_prompts, model=None, adapter=None):
    return {"activity": activity, "attr_signed": attr_signed, "attr_abs": attr_abs,
            "n_tokens": n_tokens, "n_prompts": n_prompts,
            "top_k_components": [], "per_prompt_activity": torch.zeros(n_prompts, 0),
            "prompt_tokens": [10] * n_prompts, "model": model, "adapter": adapter}


def _build_synthetic():
    """Three keys: L0.q.h0 is noise (ratio ~1, low attr); L2.k.h1 has two judging-selective
    components (high act_judge, low act_ctrl) with comp0's attribution bigger than comp1's;
    L4.gate has the single biggest raw attribution but act_ratio == 1 (excluded from candidates)."""
    activity_j = {"L0.q.h0": torch.tensor([5.0, 4.0, 6.0]),
                  "L2.k.h1": torch.tensor([80.0, 30.0]),
                  "L4.gate": torch.tensor([10.0])}
    activity_c = {"L0.q.h0": torch.tensor([5.0, 4.0, 6.0]),
                  "L2.k.h1": torch.tensor([2.0, 1.0]),
                  "L4.gate": torch.tensor([10.0])}
    attr_signed = {"L0.q.h0": torch.tensor([0.5, -0.3, 0.2]),
                   "L2.k.h1": torch.tensor([50.0, 5.0]),
                   "L4.gate": torch.tensor([-90.0])}
    attr_abs = {"L0.q.h0": torch.tensor([0.5, 0.3, 0.2]),
                "L2.k.h1": torch.tensor([50.0, 5.0]),
                "L4.gate": torch.tensor([90.0])}
    judging = _screen_result(activity_j, attr_signed, attr_abs, n_tokens=100, n_prompts=5,
                              model="Qwen3-0.6B", adapter=None)
    control = _screen_result(activity_c, {}, {}, n_tokens=100, n_prompts=5,
                              model="Qwen3-0.6B", adapter=None)
    return judging, control


def test_report_end_to_end(tmp_path, monkeypatch):
    judging, control = _build_synthetic()
    judging_path, control_path = tmp_path / "judging.pt", tmp_path / "control.pt"
    torch.save(judging, judging_path)
    torch.save(control, control_path)

    cfg = {"paths": {"out": str(tmp_path)}, "screen": {"act_eps": 1e-9, "act_ratio_min": 2.0}}
    monkeypatch.setattr(wbconfig, "cfg", lambda path=None: cfg)
    monkeypatch.setattr(sys, "argv", ["importance_report.py", "--judging", str(judging_path),
                                       "--control", str(control_path), "--tag", "t", "--top-n", "2"])
    ir.main()

    csv_path = tmp_path / "importance_t.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    # noise component and the excluded high-attr one are both < act_ratio_min
    gate_row = df[df["key"] == "L4.gate"].iloc[0]
    assert not gate_row["is_candidate"]
    noise_row = df[df["key"] == "L0.q.h0"].iloc[0]
    assert not noise_row["is_candidate"]

    # L4.gate has the single biggest raw attr_abs (90 vs 50), but is excluded from candidates
    assert gate_row["attr_abs"] > df[df["key"] == "L2.k.h1"]["attr_abs"].max()

    # the planted judging-selective component ranks first among candidates and survives the gate
    top = ir.top_components(csv_path, 2)
    assert top == [("L2.k.h1", 0), ("L2.k.h1", 1)]

    figures_dir = tmp_path / "figures"
    for stem in ("depth", "scatter", "top30"):
        assert (figures_dir / f"importance_t_{stem}.png").exists()
        assert (figures_dir / f"importance_t_{stem}.svg").exists()


def test_merge_table_raises_on_model_adapter_mismatch():
    judging, control = _build_synthetic()
    control["model"] = "Qwen3-8B"  # different model than judging's Qwen3-0.6B
    with pytest.raises(AssertionError, match="model\\+adapter mismatch"):
        ir.merge_table(judging, control, eps=1e-9, judging_path="j.pt", control_path="c.pt")


def test_merge_table_warns_instead_of_raising_on_legacy_missing_model_field(capsys):
    judging, control = _build_synthetic()
    control["model"] = None  # legacy screen result, saved before model/adapter were tracked
    control["adapter"] = None
    df = ir.merge_table(judging, control, eps=1e-9, judging_path="j.pt", control_path="c.pt")
    assert "legacy" in capsys.readouterr().out.lower()
    assert len(df) > 0  # still produces a table, just skips the identity check
