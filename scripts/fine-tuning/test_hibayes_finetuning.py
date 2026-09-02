"""Tests for the Bayesian fine-tuning GLM analysis."""


def test_checkpoint_rows_orient_style_to_the_stronger_model():
    import numpy as np
    import pandas as pd
    from hibayes_finetuning import prepare_checkpoint_rows

    pairs = pd.DataFrame([
        dict(pair_uid="a", order=0, text_1="Obviously correct", text_2="plain answer"),
        dict(pair_uid="b", order=0, text_1="plain answer", text_2="Obviously correct"),
    ])
    verdicts = pd.DataFrame([
        dict(pair_uid="a", order=0, item="i1", model_1="strong", model_2="weak",
             logit_1=2.0, logit_2=1.0),
        dict(pair_uid="b", order=0, item="i2", model_1="weak", model_2="strong",
             logit_1=2.0, logit_2=1.0),
    ])

    frame = prepare_checkpoint_rows(
        verdicts, pairs, pd.Series({"strong": 0.8, "weak": 0.5}), "qwen_base")

    assert frame.truth_side.tolist() == [1.0, -1.0]
    assert frame.is_correct.tolist() == [1, 0]
    assert np.allclose(frame.capability_gap, [0.3, 0.3])
    assert (frame.confidence_difference_stronger > 0).all()
    assert frame.judge.unique().tolist() == ["qwen_base"]


def test_paired_item_ablation_uses_items_as_replication_units():
    import numpy as np
    import pandas as pd
    from hibayes_finetuning import paired_item_ablation

    frame = pd.DataFrame({"judge": ["qwen_base"] * 4,
                          "item": ["i1", "i2", "i3", "i4"]})
    full = np.log([0.8, 0.8, 0.8, 0.8])
    reduced = np.log([0.4, 0.8, 0.4, 0.8])

    result = paired_item_ablation(full, reduced, frame, "Style").iloc[0]

    assert result.group == "Style"
    assert result.estimate == -0.5
    assert result.n_items == 4
    assert result.ci_low < result.estimate < result.ci_high


def test_posterior_comparisons_include_style_and_base_to_ft_changes():
    import numpy as np
    from hibayes_finetuning import posterior_comparisons

    judges = ["qwen_base", "qwen_ft", "llama_base", "llama_ft"]
    predictors = ["truth_side", "confidence_difference_stronger_sd", "capability_gap_sd"]
    effects = np.zeros((20, 4, 3))
    effects[:, 1, 1] = 1.0
    effects[:, 3, 1] = 2.0
    intercepts = np.zeros((20, 4))

    table = posterior_comparisons(predictors, judges, effects, intercepts)
    qwen_style = table[(table.architecture == "Qwen3-8B")
                       & (table.predictor == "Confidence")].iloc[0]

    assert "Position" in set(table.predictor)
    assert qwen_style.base_mean == 0
    assert qwen_style.ft_mean == 1
    assert qwen_style.change_mean == 1
    assert qwen_style.p_change_gt_0 == 1


def test_ablation_plot_has_all_four_groups_and_editable_text(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from matplotlib.axes import Axes
    import pandas as pd
    from hibayes_finetuning import plot_ablation

    marker_sizes = []
    original_errorbar = Axes.errorbar

    def record_errorbar(self, *args, **kwargs):
        marker_sizes.append(kwargs.get("ms"))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)

    values = {
        ("qwen_base", "Position"): -0.38,
        ("llama_base", "Position"): -0.12,
    }
    rows = []
    for judge in ("qwen_base", "qwen_ft", "llama_base", "llama_ft"):
        for i, group in enumerate(("Capability", "Length", "Style", "Position")):
            value = values.get((judge, group), -0.002 * (i + 1))
            rows.append(dict(judge=judge, group=group, estimate=value,
                             ci_low=value - 0.005, ci_high=value + 0.005, n_items=40))
    stub = tmp_path / "ablation"
    plot_ablation(pd.DataFrame(rows), stub)

    root = ET.parse(stub.with_suffix(".svg")).getroot()
    texts = ["".join(node.itertext()) for node in root.iter()
             if node.tag.rsplit("}", 1)[-1] == "text"]
    assert all(texts.count(group) == 1
               for group in ("Capability", "Length", "Style", "Position"))
    assert "Ablation Δaccuracy" in texts
    raw = stub.with_suffix(".svg").read_text()
    assert "#2a78d6" in raw and "#4a3aa7" in raw
    assert marker_sizes and all(3 <= size <= 4 for size in marker_sizes)
    assert texts.count("−0.02") == 1 and texts.count("0.01") == 1
    assert "−0.38" in texts and "−0.12" in texts
    assert "−0.4" not in texts and "0.4" not in texts
    assert float(root.attrib["height"].removesuffix("pt")) <= 155
