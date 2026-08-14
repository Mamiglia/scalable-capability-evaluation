from pool import clean_stem, match_name, sample_pairs


def test_clean_stem():
    assert clean_stem("Llama-3.1-70B-Instruct") == "llama 3.1 70b"
    assert clean_stem("claude-3-opus-20240229") == "claude 3 opus"
    assert clean_stem("Qwen3-235B-A22B-Thinking-2507") == "qwen3 235b a22b thinking 2507"


def test_match_name():
    pool = [{"name": "Llama 3.1 70B", "eci": 125.2}, {"name": "GPT-4o", "eci": 130.0},
            {"name": "Claude Opus 4.5", "eci": 141.0}, {"name": "Gemini 1.5 Pro", "eci": 126.0},
            {"name": "DeepSeek-R1", "eci": 131.0}]
    assert match_name(pool, "Llama-3.1-70B-Instruct")["eci"] == 125.2
    assert match_name(pool, "gpt-5.5_low") is None  # effort-suffixed: excluded
    assert match_name(pool, "claude-opus-4-7_xhigh") is None
    assert match_name(pool, "claude-opus-4-5-20251101")["eci"] == 141.0  # dots vs dashes
    assert match_name(pool, "gemini-1.5-pro-001")["eci"] == 126.0  # API build suffix
    assert match_name(pool, "deepseek-reasoner")["eci"] == 131.0  # API alias


def test_sample_pairs_stronger_first_and_binned():
    members = [{"short": f"m{i}", "eci": 100 + i * 4} for i in range(12)]
    pairs = sample_pairs(members, bins=[(5, 15), (15, 25), (25, 40)], per_bin=3, seed=0)
    assert len(pairs) == 9
    for a, b in pairs:
        assert a["eci"] > b["eci"]
        assert 5 <= a["eci"] - b["eci"] < 40
