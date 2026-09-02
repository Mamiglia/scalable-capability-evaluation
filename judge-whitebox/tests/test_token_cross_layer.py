import numpy as np

import token_cross_layer as cross


def synthetic_layers(n_items=60, prompts_per_item=2, seed=0, novel=False):
    rng = np.random.default_rng(seed)
    n = n_items * prompts_per_item
    item = np.repeat(np.arange(n_items), prompts_per_item)
    early = rng.normal(size=(n, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if novel:
        gap = rng.normal(size=n)
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        late = early @ rotation + 3 * gap[:, None] * direction
    else:
        gap = early[:, 0] + .05 * rng.normal(size=n)
        late = early @ rotation + .02 * rng.normal(size=(n, 4))
    return early.astype(np.float32), late.astype(np.float32), gap, item


def test_cross_layer_decomposition_recognises_inherited_rotation():
    early, late, gap, item = synthetic_layers(novel=False)

    out = cross.decompose(early, late, gap, item, cv_folds=3)

    assert out.map_late_r2.mean() > .9
    assert out.gap_inherited.mean() > .7
    assert abs(out.novel_gain.mean()) < .02


def test_cross_layer_decomposition_finds_new_late_gap_information():
    early, late, gap, item = synthetic_layers(seed=1, novel=True)

    out = cross.decompose(early, late, gap, item, cv_folds=3)

    assert out.gap_early.mean() < .1
    assert out.gap_novel_residual.mean() > .5
    assert out.novel_gain.mean() > .5
