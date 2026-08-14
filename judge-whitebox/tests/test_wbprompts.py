import wbprompts
from configs.experiment import SYSTEM_PROMPT


def test_short_prompt_is_full_minus_explanation():
    # every paragraph of SHORT except the reply-format block appears in src's prompt
    for para in wbprompts.SYSTEM_PROMPT_SHORT.split("\n\n")[:2]:
        assert para in SYSTEM_PROMPT


def test_render_assembles_messages():
    class T:  # stub tokenizer
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert [m["role"] for m in msgs] == ["system", "user"]
            return "SYS:" + msgs[0]["content"][:20] + "|USR:" + msgs[1]["content"][:20]
    row = dict(question="Q?", options_1="", options_2="", text_1="t1", text_2="t2")
    out = wbprompts.render_verdict_prompt(row, T())
    assert out.endswith("ANSWER: ") and "USR:" in out


def test_verbatim_against_judge_ft_if_present():
    import pathlib
    src = pathlib.Path("../judge-ft/scripts/fine-tuning/prompts.py")
    if src.exists():
        assert wbprompts.SYSTEM_PROMPT_SHORT in src.read_text()


def test_weaker_constants_pin_to_originals():
    # WEAKER constants must differ from the originals ONLY in "less capable" vs
    # "more capable" (case-preserving: the ask sentence uses "LESS", the answer
    # format line uses "less") - reversing that substitution must reproduce the
    # original exactly, pinning the two pairs together against drift.
    restored_system = wbprompts.SYSTEM_PROMPT_SHORT_WEAKER \
        .replace("LESS capable", "more capable").replace("less capable", "more capable")
    assert restored_system == wbprompts.SYSTEM_PROMPT_SHORT
    assert wbprompts.SYSTEM_PROMPT_SHORT_WEAKER != wbprompts.SYSTEM_PROMPT_SHORT
    assert wbprompts.USER_PROMPT_WEAKER.replace("less capable", "more capable") \
        == wbprompts.USER_PROMPT
    assert wbprompts.USER_PROMPT_WEAKER != wbprompts.USER_PROMPT


def test_render_not_inverted_unchanged():
    class T:  # stub tokenizer
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert [m["role"] for m in msgs] == ["system", "user"]
            return msgs[0]["content"] + "|" + msgs[1]["content"]
    row = dict(question="Q?", options_1="", options_2="", text_1="t1", text_2="t2")
    out = wbprompts.render_verdict_prompt(row, T())
    assert out.endswith("ANSWER: ")
    assert wbprompts.SYSTEM_PROMPT_SHORT in out
    assert "more capable model (1 or 2)" in out
    assert "from the more capable model?" in out
    assert "less capable" not in out


def test_render_inverted_swaps_both_surfaces():
    class T:  # stub tokenizer
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            assert [m["role"] for m in msgs] == ["system", "user"]
            return msgs[0]["content"] + "|" + msgs[1]["content"]
    row = dict(question="Q?", options_1="", options_2="", text_1="t1", text_2="t2")
    out = wbprompts.render_verdict_prompt(row, T(), invert=True)
    assert out.endswith("ANSWER: ")
    # both surfaces (system + closing line) inverted
    assert wbprompts.SYSTEM_PROMPT_SHORT_WEAKER in out
    assert "LESS capable model (1 or 2)" in out
    assert "from the less capable model?" in out
    # no "more capable" ask survives in either surface
    assert "more capable model (1 or 2)" not in out
    assert "from the more capable model?" not in out
