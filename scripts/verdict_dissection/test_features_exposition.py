import json

from features_exposition import exposition_of


def parts(text):
    return json.dumps([{"role": "assistant", "subparts": [{"kind": "content", "content": text}]}])


def test_exposition_counts():
    text = ("Let's see. The closest option is B, roughly 20 GeV. However, if we use m = 6u, hmm.\n"
            "**Total energy:** E = γmc² ⇒ 20.13 GeV\nStep 2: therefore the answer is C.\n\nANSWER: C")
    f = exposition_of(parts(text))
    assert f["think_aloud_per_100_words"] > 0 and f["option_matching_per_100_words"] > 0
    assert f["conditionals_per_100_words"] > 0 and f["consequence_per_100_words"] > 0
    assert f["step_headers"] == 1 and f["label_line_share"] == 1 / 3 and f["bold_labels_per_100_words"] > 0
