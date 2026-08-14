"""Tests for the judge fine-tuning scripts."""


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
