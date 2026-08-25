import torch

from sae_feature_traces import boundaries, query_mask


def test_boundaries_thin_to_limit_and_keep_end():
    assert boundaries(5, 512) == [1, 2, 3, 4, 5]
    thinned = boundaries(10_000, 512)
    assert len(thinned) <= 512 and thinned[-1] == 10_000 and thinned == sorted(set(thinned))


def test_query_mask_sees_own_prefix_and_self():
    mask = query_mask(4, [1, 4])
    assert mask.shape == (2, 6)
    assert mask[0].tolist() == [True, False, False, False, True, False]
    assert mask[1].tolist() == [True, True, True, True, False, True]
    assert mask.dtype == torch.bool
