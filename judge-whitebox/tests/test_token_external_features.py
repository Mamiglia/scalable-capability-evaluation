import numpy as np
import pandas as pd

import token_external_features as features


def test_external_frame_identifies_longer_weaker_error_from_exact_token_counts():
    prompts = [
        {"pair_uid": "p0", "order": 0, "item": "i0", "gap": 2.0},
        {"pair_uid": "p0", "order": 1, "item": "i0", "gap": 1.0},
    ]
    pairs = pd.DataFrame([
        {"pair_uid": "p0", "order": 0, "cap_1": .8, "cap_2": .7, "label": 1,
         "text_1": "long strong answer", "text_2": "short answer"},
        {"pair_uid": "p0", "order": 1, "cap_1": .7, "cap_2": .8, "label": 2,
         "text_1": "long weak answer", "text_2": "short answer"},
    ])
    prompt_idx = np.repeat([0, 1], 8)
    group_id = np.tile([3, 3, 3, 3, 3, 4, 4, 4], 2)

    out = features.external_frame(prompts, pairs, prompt_idx, group_id)

    assert out.transcript_1_tokens.tolist() == [5, 5]
    assert out.transcript_2_tokens.tolist() == [3, 3]
    assert out.pick.tolist() == [1, 1]
    assert out.correct.tolist() == [True, False]
    assert out.longer_weaker.tolist() == [False, True]
    assert out.picked_longer_weaker.tolist() == [False, True]
    assert out.capability_gap.tolist() == [.1, -.1]


def test_external_frame_marks_length_ties_without_assigning_a_longer_slot():
    prompts = [{"pair_uid": "p0", "order": 0, "item": "i0", "gap": -1.0}]
    pairs = pd.DataFrame([{
        "pair_uid": "p0", "order": 0, "cap_1": .8, "cap_2": .7, "label": 1,
        "text_1": "a", "text_2": "b",
    }])

    out = features.external_frame(prompts, pairs, np.zeros(4, dtype=int),
                                  np.array([3, 3, 4, 4]))

    assert out.longer_slot.tolist() == [0]
    assert out.picked_longer.tolist() == [False]
