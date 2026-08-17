"""Minimal tensor-native BatchTopK SAE.

Derived from the MIT-licensed reference implementation in
https://github.com/bartbussmann/BatchTopK at
b9aab1c6156381ae7ae2997e3490e7b99e195dde (Bussmann, Leask, Nanda, 2024).
This keeps only the training/inference pieces needed for already-recorded
activation tensors; it does not include the upstream TransformerLens runner.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BatchTopKSAE(nn.Module):
    """BatchTopK training SAE with fixed-threshold, batch-independent inference."""

    def __init__(self, d_in, d_sae, k, *, dead_after=5, aux_k=512, aux_penalty=1 / 32):
        super().__init__()
        if not 0 < k <= d_sae:
            raise ValueError("k must be in [1, d_sae]")
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.dead_after, self.aux_k, self.aux_penalty = dead_after, aux_k, aux_penalty
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("inactive_batches", torch.zeros(d_sae))
        self.register_buffer("threshold", torch.tensor(float("nan")))
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _pre_activations(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc)

    def encode_training(self, x):
        """Select exactly ``len(x) * k`` activations globally over this batch."""
        acts = self._pre_activations(x)
        chosen = acts.flatten().topk(len(x) * self.k)
        return torch.zeros_like(acts.flatten()).scatter_(0, chosen.indices, chosen.values).view_as(acts)

    def decode(self, codes):
        return codes @ self.W_dec + self.b_dec

    def training_loss(self, x):
        """Return BatchTopK loss and tensors/diagnostics for one optimizer step."""
        acts = self._pre_activations(x)
        chosen = acts.flatten().topk(len(x) * self.k)
        codes = torch.zeros_like(acts.flatten()).scatter_(0, chosen.indices, chosen.values).view_as(acts)
        reconstruction = self.decode(codes)
        self._update_inactive(codes)
        aux_loss = self._auxiliary_loss(x, reconstruction, acts)
        reconstruction_loss = F.mse_loss(reconstruction.float(), x.float())
        loss = reconstruction_loss + aux_loss
        return loss, {"reconstruction": reconstruction, "codes": codes,
                      "reconstruction_loss": reconstruction_loss.detach(),
                      "aux_loss": aux_loss,
                      "mean_l0": (codes > 0).float().sum(1).mean().detach()}

    @torch.no_grad()
    def _update_inactive(self, codes):
        active = codes.sum(0) > 0
        self.inactive_batches.add_((~active).to(self.inactive_batches))
        self.inactive_batches.masked_fill_(active, 0)

    def _auxiliary_loss(self, x, reconstruction, acts):
        dead = self.inactive_batches >= self.dead_after
        if not dead.any():
            return x.new_zeros(())
        n_aux = min(self.aux_k, int(dead.sum()))
        selected = acts[:, dead].topk(n_aux, dim=-1)
        aux_codes = torch.zeros_like(acts[:, dead]).scatter_(-1, selected.indices, selected.values)
        residual = x.float() - reconstruction.float()
        return self.aux_penalty * F.mse_loss(aux_codes @ self.W_dec[dead], residual)

    @torch.no_grad()
    def calibrate_threshold(self, batches):
        """Calibrate the paper's batch-independent inference threshold."""
        minima = []
        for batch in batches:
            codes = self.encode_training(batch)
            positive = codes[codes > 0]
            if positive.numel():
                minima.append(positive.min())
        if not minima:
            raise ValueError("calibration batches contained no positive activations")
        self.threshold.copy_(torch.stack(minima).mean())
        return self.threshold

    def encode(self, x):
        """Encode with the calibrated global threshold, independent of batch composition."""
        if torch.isnan(self.threshold):
            raise RuntimeError("call calibrate_threshold before inference")
        acts = self._pre_activations(x)
        return torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        codes = self.encode(x)
        return self.decode(codes), codes

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_decoder_grad_and_normalize(self):
        """Project decoder gradients tangent to the unit-norm decoder constraint."""
        self._normalize_decoder()
        if self.W_dec.grad is not None:
            self.W_dec.grad.sub_((self.W_dec.grad * self.W_dec).sum(-1, keepdim=True) * self.W_dec)
