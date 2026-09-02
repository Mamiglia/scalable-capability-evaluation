import pandas as pd
import corpus


def make_pairs():
    rows = [
        {"pair_uid": "p1", "order": 0, "question": "What is 2+2?",
         "text_1": "The answer is 4.", "text_2": "It is four."},
        {"pair_uid": "p1", "order": 1, "question": "What is 2+2?",
         "text_1": "It is four.", "text_2": "The answer is 4."},
        {"pair_uid": "p2", "order": 0, "question": "Name a color.",
         "text_1": "Blue.", "text_2": "Red."},
        {"pair_uid": "p2", "order": 1, "question": "Name a color.",
         "text_1": "Red.", "text_2": "Blue."},
    ]
    return pd.DataFrame(rows)


def test_two_kinds_per_transcript_no_limit():
    out = corpus.build_control(make_pairs())
    # 2 pairs x 2 transcripts x 2 kinds
    assert len(out) == 8
    kinds = [e["kind"] for e in out]
    assert kinds.count("summarise") == 4
    assert kinds.count("continuation") == 4


def test_limit_caps_pairs_not_rows():
    out = corpus.build_control(make_pairs(), limit=1)
    assert len(out) == 4  # 1 pair x 2 transcripts x 2 kinds
    texts = [e.get("text", "") for e in out if e["kind"] == "continuation"]
    assert all("2+2" in t for t in texts)  # only p1 kept


def test_dedupes_repeated_transcript_text_across_pairs():
    rows = [
        {"pair_uid": "p1", "order": 0, "question": "What is 2+2?",
         "text_1": "The answer is 4.", "text_2": "It is four."},
        {"pair_uid": "p1", "order": 1, "question": "What is 2+2?",
         "text_1": "It is four.", "text_2": "The answer is 4."},
        # p3 reuses p1's exact text_1 ("The answer is 4.") against a different opponent transcript
        {"pair_uid": "p3", "order": 0, "question": "What is two plus two?",
         "text_1": "The answer is 4.", "text_2": "Something else."},
        {"pair_uid": "p3", "order": 1, "question": "What is two plus two?",
         "text_1": "Something else.", "text_2": "The answer is 4."},
    ]
    df = pd.DataFrame(rows)
    out = corpus.build_control(df)
    # 3 unique transcript texts ("The answer is 4.", "It is four.", "Something else.") x 2 kinds
    assert len(out) == 6
    continuation_texts = [e["text"] for e in out if e["kind"] == "continuation"]
    assert sum("The answer is 4." in t for t in continuation_texts) == 1


def test_no_judging_vocabulary():
    out = corpus.build_control(make_pairs())
    for entry in out:
        blob = str(entry)
        assert "capable" not in blob
        assert "Transcript 1" not in blob
        assert "ANSWER" not in blob


def test_summarise_contains_transcript_as_chat_frame():
    out = corpus.build_control(make_pairs())
    summarise = [e for e in out if e["kind"] == "summarise"][0]
    assert summarise["messages"][0]["role"] == "system"
    assert summarise["messages"][1]["role"] == "user"
    assert "The answer is 4." in summarise["messages"][1]["content"] or \
           "It is four." in summarise["messages"][1]["content"]


def test_continuation_has_no_chat_template():
    out = corpus.build_control(make_pairs())
    continuation = [e for e in out if e["kind"] == "continuation"][0]
    assert set(continuation.keys()) == {"kind", "text"}
    assert isinstance(continuation["text"], str)
    assert "role" not in continuation["text"]
