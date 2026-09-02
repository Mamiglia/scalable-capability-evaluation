"""Tests for the judge fine-tuning scripts."""


def test_prepare_rows_uses_directional_accuracy_and_log_length_gaps():
    import pandas as pd
    from length_bias_glm import prepare_rows

    pairs = pd.DataFrame([dict(
        pair_uid="p", order=0, item="i", model_1="strong", model_2="weak",
        text_1="one two three four five six seven eight", text_2="one two",
    )])
    verdicts = pd.DataFrame([dict(
        pair_uid="p", order=0, item="i", model_1="strong", model_2="weak",
        logit_1=2.0, logit_2=1.0,
    )])
    rows = prepare_rows(
        verdicts, pairs, pd.Series({"strong": 0.7, "weak": 0.5}), fine_tuned=True)

    assert rows.loc[0, "choose_1"] == 1
    assert abs(rows.loc[0, "accuracy_gap"] - 2.0) < 1e-12  # per 10 percentage points
    assert abs(rows.loc[0, "length_gap"] - 2.0) < 1e-12    # log2(8 / 2)
    assert rows.loc[0, "fine_tuned"] == 1


def test_clustered_glm_recovers_checkpoint_specific_length_effects():
    import numpy as np
    import pandas as pd
    from length_bias_glm import fit_clustered_logit, summarize_fit

    rng = np.random.default_rng(7)
    n = 12_000
    frame = pd.DataFrame({
        "item": np.repeat(np.arange(120), n // 120),
        "accuracy_gap": rng.normal(size=n),
        "length_gap": rng.normal(size=n),
        "fine_tuned": rng.integers(0, 2, size=n),
    })
    x = np.column_stack([
        np.ones(n), frame.accuracy_gap, frame.length_gap, frame.fine_tuned,
        frame.accuracy_gap * frame.fine_tuned,
        frame.length_gap * frame.fine_tuned,
    ])
    truth = np.array([0.1, 0.7, 0.6, -0.2, 0.4, -0.8])
    frame["choose_1"] = rng.binomial(1, 1 / (1 + np.exp(-(x @ truth))))

    beta, cov = fit_clustered_logit(frame)
    table = summarize_fit(beta, cov, "test judge", len(frame), frame.item.nunique())
    length = table[table.effect == "Length (2× longer)"].iloc[0]

    assert np.allclose(beta, truth, atol=0.12)
    assert abs(length.base_estimate - truth[2]) < 0.12
    assert abs(length.ft_estimate - (truth[2] + truth[5])) < 0.12
    assert length.change < -0.6
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_coefficient_plot_matches_main_glm_orientation(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from matplotlib.axes import Axes
    import pandas as pd
    from length_bias_glm import REGRESSORS, plot_coefficients

    errorbars = []
    original_errorbar = Axes.errorbar

    def record_errorbar(self, *args, **kwargs):
        errorbars.append(kwargs.copy())
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", record_errorbar)

    rows = []
    for judge in ("Qwen3-8B", "Llama-3.2-3B"):
        for index, (predictor, effect, group) in enumerate(REGRESSORS):
            value = -3.0 if predictor == "truth_side" else 0.05 * index
            rows.append(dict(judge=judge, predictor=predictor, effect=effect, group=group,
                             base_estimate=value, base_ci_low=value - 0.2,
                             base_ci_high=value + 0.2, ft_estimate=value / 2,
                             ft_ci_low=value / 2 - 0.2, ft_ci_high=value / 2 + 0.2))
    stub = tmp_path / "coefficients"
    plot_coefficients(pd.DataFrame(rows), stub)

    root = ET.parse(stub.with_suffix(".svg")).getroot()
    texts = ["".join(node.itertext()) for node in root.iter()
             if node.tag.rsplit("}", 1)[-1] == "text"]
    width = float(root.attrib["width"].removesuffix("pt"))
    height = float(root.attrib["height"].removesuffix("pt"))
    labels = [label for _, label, _ in REGRESSORS]
    assert all(texts.count(label) >= 1 for label in labels)
    assert texts.count("Style") == 1
    assert labels[0] == "Capability gap" and labels.index("Position") < labels.index("Pronouns")
    label_nodes = [node for node in root.iter()
                   if node.tag.rsplit("}", 1)[-1] == "text"]
    assert all(any("".join(node.itertext()) == label and
                       "rotate(-90" in node.attrib.get("transform", "")
                       for node in label_nodes) for label in labels)
    assert texts.count("−3.5") == 1 and texts.count("3.5") == 1
    assert "Coefficient (log-odds)" in texts
    legend_labels = ("Qwen base", "Qwen FT", "Llama base", "Llama FT")
    assert all(label in texts for label in legend_labels)
    raw = stub.with_suffix(".svg").read_text()
    assert "#2a78d6" in raw and "#4a3aa7" in raw
    assert len(errorbars) == 4 * len(REGRESSORS)
    assert all(call.get("barsabove") is True and call.get("capsize") == 0
               and call.get("elinewidth") == 1 and 3 <= call.get("ms") <= 4
               for call in errorbars)
    assert 308 <= width <= 322
    assert width / height > 1.6
    assert height <= 205


def test_full_glm_results_markdown_includes_estimates_changes_and_p_values():
    import pandas as pd
    from length_bias_glm import format_results_markdown

    table = pd.DataFrame([dict(
        judge="Qwen3-8B", effect="Length (2× longer)",
        base_estimate=0.10, base_ci_low=0.01, base_ci_high=0.18,
        ft_estimate=-0.07, ft_ci_low=-0.16, ft_ci_high=0.02,
        change=-0.17, change_ci_low=-0.26, change_ci_high=-0.08,
        p_change=0.00058, n_rows=18000, n_items=40,
    )])

    result = format_results_markdown(table)

    assert "Base β [95% CI]" in result
    assert "Fine-tuned β [95% CI]" in result
    assert "Δβ [95% CI]" in result and "p(Δ)" in result
    assert "+0.100 [+0.010, +0.180]" in result
    assert "5.80e-04" in result
    assert "18,000 verdicts" in result and "40 held-out items" in result
    assert "Hierarchical Bayesian" in result and "95% credible intervals" in result
    assert "Unpenalized" not in result


def test_prompts_match_src():
    from configs.experiment import SYSTEM_PROMPT
    from judge.dataset import USER_PROMPT, OPTIONS_BLOCK
    import prompts
    assert prompts.SYSTEM_PROMPT_FULL == SYSTEM_PROMPT
    assert prompts.USER_PROMPT == USER_PROMPT
    assert prompts.OPTIONS_BLOCK == OPTIONS_BLOCK
    # short variant: full with the 3-line reply format reduced to ANSWER only
    assert "EXPLANATION" not in prompts.SYSTEM_PROMPT_SHORT
    assert "ANSWER: <1 or 2" in prompts.SYSTEM_PROMPT_SHORT
    assert prompts.SYSTEM_PROMPT_SHORT != prompts.SYSTEM_PROMPT_FULL


def test_format_options():
    import prompts
    assert prompts.format_options([], 1) == ""
    out = prompts.format_options(["x", "y"], 2)
    assert "Model 2" in out and "A) x" in out and "B) y" in out


def test_base_group():
    from prep_pairs import base_group
    assert base_group("claude-opus-4-5-20251101_16K") == "claude-opus-4-5-20251101"
    assert base_group("gemini-3.5-flash_high") == "gemini-3.5-flash"
    assert base_group("DeepSeek-V3") == "DeepSeek-V3"


def test_has_reasoning():
    from prep_pairs import has_reasoning
    assert not has_reasoning("ANSWER: C", min_chars=100)              # answer-only
    assert not has_reasoning("short\nANSWER: C", min_chars=100)       # too little reasoning
    assert has_reasoning("x" * 100 + "\nANSWER: C", min_chars=100)    # real reasoning


def test_make_pair_rows_counterbalanced():
    from prep_pairs import make_pair_rows, check_counterbalance
    import pandas as pd
    a = dict(model="strong", cap=2.0, text="ta", options="oa")
    b = dict(model="weak", cap=1.0, text="tb", options="ob")
    rows = make_pair_rows("item1", a, b, question="q")
    df = pd.DataFrame(rows)
    assert len(df) == 2
    assert set(df.label) == {1, 2}
    r1, r2 = df.iloc[0], df.iloc[1]
    assert r1.text_1 == r2.text_2 and r1.text_2 == r2.text_1
    assert (r1.label == 1) == (r1.model_1 == "strong")
    df["split"] = "train"
    check_counterbalance(df)  # must not raise


def test_check_counterbalance_catches_imbalance():
    from prep_pairs import make_pair_rows, check_counterbalance
    import pandas as pd, pytest
    a = dict(model="strong", cap=2.0, text="ta", options="oa")
    b = dict(model="weak", cap=1.0, text="tb", options="ob")
    df = pd.DataFrame(make_pair_rows("i", a, b, question="q")).iloc[:1]
    df["split"] = "train"
    with pytest.raises(AssertionError):
        check_counterbalance(df)


def test_assign_splits():
    from prep_pairs import assign_splits
    import pandas as pd
    df = pd.DataFrame([
        dict(model_1="hm", model_2="hn", item="hi"),   # both held-out models, held-out item
        dict(model_1="hm", model_2="t1", item="hi"),   # mixed, held-out item
        dict(model_1="t1", model_2="t2", item="hi"),   # train models, held-out item
        dict(model_1="t1", model_2="t2", item="ti"),   # train models, train item
        dict(model_1="hm", model_2="t1", item="ti"),   # held-out model on train item: excluded
    ])
    s = assign_splits(df, holdout_models={"hm", "hn"}, holdout_items={"hi"})
    assert list(s) == ["eval_mm", "eval_mixed", "eval_items", "train", "drop"]


def _fake_verdicts(elos, n_items=60, pos=0.5, seed=0):
    # synthetic pairs among models with true elos; picked_1 ~ sigmoid(gap + pos)
    import numpy as np, pandas as pd, itertools
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for m1, m2 in itertools.permutations(elos, 2):
            p = 1 / (1 + np.exp(-(elos[m1] - elos[m2] + pos)))
            l1 = float(rng.random() < p)
            rows.append(dict(pair_uid=f"{i}|{min(m1,m2)}|{max(m1,m2)}", order=0,
                             item=str(i), model_1=m1, model_2=m2,
                             label=1 if elos[m1] > elos[m2] else 2,
                             logit_1=l1, logit_2=1 - l1))
    return pd.DataFrame(rows)


def test_bt_recovers_order_and_position():
    from analysis import bt_fit
    elos = {"a": 1.5, "b": 0.5, "c": -0.5, "d": -1.5}
    df = _fake_verdicts(elos)
    fit, pos = bt_fit(df, c=1.0)
    assert list(fit.sort_values(ascending=False).index) == ["a", "b", "c", "d"]
    assert pos > 0.2


def test_accuracy_by_split():
    from analysis import accuracy_by_split
    import pandas as pd
    df = pd.DataFrame([dict(split="eval_mm", label=1, logit_1=2.0, logit_2=1.0, pair_uid="p", order=0),
                       dict(split="eval_mm", label=2, logit_1=1.0, logit_2=2.0, pair_uid="p", order=1)])
    acc = accuracy_by_split(df)
    assert acc["eval_mm"] == 1.0


def test_loss_plateaued():
    from train import loss_plateaued
    flat = [0.69] * 200
    assert not loss_plateaued(flat, step=500, check_steps=1000, threshold=0.65)   # grace period
    assert loss_plateaued(flat, step=1500, check_steps=1000, threshold=0.65)      # pinned at ln 2
    assert not loss_plateaued([0.3] * 200, step=1500, check_steps=1000, threshold=0.65)


def test_render_example_masks_all_but_verdict():
    from train import render_example
    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    row = pd.Series(dict(question="q?", options_1="", options_2="",
                         text_1="alpha", text_2="beta", label=2))
    ids, pos = render_example(row, tok, max_input_tokens=4096)
    assert pos == len(ids) - 1
    assert tok.decode([ids[pos]]) == "2"
    assert "ANSWER:" in tok.decode(ids[-8:])
