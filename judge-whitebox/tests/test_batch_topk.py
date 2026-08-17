import torch

from batch_topk import BatchTopKSAE


def test_training_codes_use_one_global_batch_budget_with_variable_row_l0():
    sae = BatchTopKSAE(d_in=2, d_sae=4, k=2)
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1., 2., 3., 4.], [0., 0., 0., 0.]]))
        sae.b_dec.zero_()
    x = torch.tensor([[1., 0.], [.8, 0.], [.2, 0.]])

    codes = sae.encode_training(x)

    assert int((codes > 0).sum()) == len(x) * sae.k
    assert torch.equal((codes > 0).sum(1), torch.tensor([3, 3, 0]))


def test_calibrated_threshold_encoding_is_invariant_to_batch_composition():
    sae = BatchTopKSAE(d_in=2, d_sae=3, k=1)
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1., 2., 3.], [0., 0., 0.]]))
        sae.W_dec.copy_(torch.eye(3, 2))
        sae.b_dec.zero_()
    sae.calibrate_threshold([torch.tensor([[1., 0.], [.5, 0.]])])
    target = torch.tensor([[.75, 0.]])

    alone = sae.encode(target)
    among_other_tokens = sae.encode(torch.cat([target, torch.tensor([[9., 0.], [.1, 0.]])]))[:1]

    assert torch.equal(alone, among_other_tokens)
    assert torch.equal(sae.reconstruct(target), sae.reconstruct(torch.cat([
        target, torch.tensor([[9., 0.], [.1, 0.]])
    ]))[:1])


def test_dead_feature_aux_loss_backpropagates_without_resampling_weights():
    torch.manual_seed(0)
    sae = BatchTopKSAE(d_in=3, d_sae=6, k=1, dead_after=1, aux_k=2, aux_penalty=1.)
    with torch.no_grad():
        sae.W_enc[:, :2].fill_(1.)
    sae.inactive_batches[:2] = 1
    before = sae.W_dec.detach().clone()

    _, metrics = sae.training_loss(torch.ones(5, 3))
    metrics["aux_loss"].backward()

    assert metrics["aux_loss"] > 0
    assert sae.W_enc.grad[:, :2].abs().sum() > 0
    assert torch.equal(sae.W_dec, before)


def test_decoder_gradient_projection_is_tangent_and_normalizes_rows():
    torch.manual_seed(1)
    sae = BatchTopKSAE(d_in=3, d_sae=5, k=1)
    (sae.W_dec.square().sum()).backward()

    sae.project_decoder_grad_and_normalize()

    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose((sae.W_dec.grad * sae.W_dec).sum(1), torch.zeros(5), atol=1e-6)


def test_batch_topk_learns_a_small_sparse_mixture():
    torch.manual_seed(2)
    dictionary = torch.randn(4, 6)
    codes = torch.zeros(96, 4)
    indices = torch.randint(0, 4, (96,))
    codes[torch.arange(96), indices] = torch.rand(96) + .5
    x = codes @ dictionary
    sae = BatchTopKSAE(d_in=6, d_sae=8, k=1, dead_after=1000)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-3)
    initial, _ = sae.training_loss(x)
    for _ in range(180):
        loss, _ = sae.training_loss(x)
        optimizer.zero_grad()
        loss.backward()
        sae.project_decoder_grad_and_normalize()
        optimizer.step()
    final, _ = sae.training_loss(x)

    assert final < initial * .5
