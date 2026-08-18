"""Token attribution to a judge's answer-logit gap."""

import torch

import wbprompts


def answer_token_ids(tokenizer):
    """Single token ids for the two answer options."""
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in wbprompts.ANSWER_TOKENS]
    assert all(len(i) == 1 for i in ids), f"answer tokens must be single-token: {ids}"
    (id1,), (id2,) = ids
    return id1, id2


def _gap_logit(model, embeds, attention_mask, id1, id2):
    """Differentiable gap from input embeddings."""
    out = model(inputs_embeds=embeds, attention_mask=attention_mask)
    return out.logits[0, -1, id1] - out.logits[0, -1, id2]


def attribute_tokens(model, tokenizer, input_ids, attention_mask, steps):
    """Integrated-gradient attribution of the gap to each input token."""
    assert steps > 0
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    id1, id2 = answer_token_ids(tokenizer)

    embeds = model.get_input_embeddings()(input_ids).detach()
    baseline = torch.zeros_like(embeds)
    delta = embeds - baseline

    total = torch.zeros_like(embeds)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (baseline + alpha * delta).detach().requires_grad_(True)
        gap = _gap_logit(model, point, attention_mask, id1, id2)
        grad, = torch.autograd.grad(gap, point)
        total += grad

    attributions = (total / steps) * delta
    scores = attributions.sum(-1)[0].double().cpu().numpy()

    with torch.no_grad():
        gap = float(_gap_logit(model, embeds, attention_mask, id1, id2))
        baseline_gap = float(_gap_logit(model, baseline, attention_mask, id1, id2))
    return {"scores": scores, "gap": gap, "baseline_gap": baseline_gap,
            "completeness_error": float(scores.sum() - (gap - baseline_gap))}
