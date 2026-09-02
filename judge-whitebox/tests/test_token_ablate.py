import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tokenizer is already cached locally, no network

import token_ablate
import wbconfig
import wbprompts

STRUCTURAL_GROUPS = ["instr_format", "slot_labels", "question", "transcript_1", "transcript_2",
                     "answer_readout"]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("Qwen/Qwen3-0.6B tokenizer not cached locally")


@pytest.fixture
def row():
    return dict(
        question="What is 2+2?", options_1="", options_2="",
        text_1="The answer is four. Therefore, the calculation is correct.",
        text_2="Four is the answer, since two plus two equals four.",
    )


@pytest.fixture(scope="module")
def c():
    return wbconfig.cfg()


def test_structural_groups_disjoint_and_cover_all_tokens(tokenizer, row, c):
    groups = token_ablate.token_groups(row, tokenizer, c)
    n = len(tokenizer(wbprompts.render_verdict_prompt(row, tokenizer), return_offsets_mapping=True)
             ["input_ids"])

    seen = set()
    for name in STRUCTURAL_GROUPS:
        idx = set(groups[name])
        assert not (idx & seen), f"{name} overlaps an earlier structural group"
        seen |= idx
    assert seen <= set(range(n))


def test_transcript_spans_cover_their_own_text(tokenizer, row, c):
    groups = token_ablate.token_groups(row, tokenizer, c)
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    start1 = prompt.find(row["text_1"])
    end1 = start1 + len(row["text_1"])
    start2 = prompt.find(row["text_2"], end1)
    end2 = start2 + len(row["text_2"])

    for name, (start, end) in (("transcript_1", (start1, end1)), ("transcript_2", (start2, end2))):
        idx = groups[name]
        assert idx, name
        # every returned token overlaps the span
        assert all(offsets[i][0] < end and offsets[i][1] > start for i in idx)
        # the span's interior chars are all covered (skip a couple of boundary chars, which
        # may fall to a neighbouring token depending on the tokenizer's merges)
        covered = set()
        for i in idx:
            covered.update(range(offsets[i][0], offsets[i][1]))
        assert set(range(start + 2, end - 2)) <= covered


def test_answer_readout_is_last_token(tokenizer, row, c):
    groups = token_ablate.token_groups(row, tokenizer, c)
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    n = len(tokenizer(prompt, return_offsets_mapping=True)["input_ids"])
    assert groups["answer_readout"] == [n - 1]


def test_connective_marker_hits_therefore_in_text_1(tokenizer, row, c):
    groups = token_ablate.token_groups(row, tokenizer, c)
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]

    start1 = prompt.find(row["text_1"])
    therefore_char = start1 + row["text_1"].index("Therefore")
    therefore_tokens = {i for i, (s, e) in enumerate(offsets)
                        if s < therefore_char + len("Therefore") and e > therefore_char}

    assert therefore_tokens & set(groups["connective"])
    assert set(groups["connective"]) <= set(groups["transcript_1"]) | set(groups["transcript_2"])


def test_no_marker_hit_for_text_without_lexicon_words(tokenizer, c):
    row = dict(question="Q?", options_1="", options_2="",
               text_1="A plain sentence about cats and dogs playing outside together.",
               text_2="A calm sentence describing a quiet afternoon in the park.")
    groups = token_ablate.token_groups(row, tokenizer, c)
    assert groups["connective"] == []
    assert groups["hedge"] == []
    assert groups["backtrack"] == []


def test_causal_mean_uses_only_earlier_kept_positions():
    """_causal_mean replaces each ablated position with the mean over KEPT positions strictly
    earlier than it, never leaking future-token content backwards."""
    torch = pytest.importorskip("torch")
    hs = torch.arange(5.0).view(1, 5, 1)          # value == position index
    out = token_ablate._causal_mean(hs, [2, 4])   # ablate 2 and 4; kept = {0,1,3}
    assert abs(out[0, 2, 0].item() - 0.5) < 1e-6      # mean{0,1}
    assert abs(out[0, 4, 0].item() - 4 / 3) < 1e-6    # mean{0,1,3}
    for p in (0, 1, 3):                                # kept positions unchanged
        assert out[0, p, 0].item() == p
    # future-leak guard: changing later positions must not move an earlier ablated position
    hs2 = hs.clone(); hs2[0, 3, 0] = 99; hs2[0, 4, 0] = 99
    assert abs(token_ablate._causal_mean(hs2, [2, 4])[0, 2, 0].item() - 0.5) < 1e-6
    # leading span (no earlier kept position) is zeroed, not left intact
    lead = torch.arange(5.0).view(1, 5, 1) + 10          # position 0 == 10, not 0
    assert token_ablate._causal_mean(lead, [0])[0, 0, 0].item() == 0
    out01 = token_ablate._causal_mean(lead, [0, 1])       # both leading -> both zeroed
    assert out01[0, 0, 0].item() == 0 and out01[0, 1, 0].item() == 0
