import torch
from transformers import AutoModelForCausalLM, Qwen3Config

import token_attribution


def make_tiny_model():
    """Random-weight Qwen3 small enough for a CPU forward pass.

    Seeded: how many integration steps completeness needs depends on the draw.
    """
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=4, max_position_embeddings=32)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


class DigitTokenizer:
    """Maps "1" and "2" to single ids so the answer-token assertion holds."""

    def encode(self, text, add_special_tokens=False):
        return [{"1": 1, "2": 2}[text]]


def test_attributions_sum_to_the_gap_difference():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)

    result = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=1024)

    assert result["scores"].shape == (6,)
    span = abs(result["gap"] - result["baseline_gap"])
    assert abs(result["completeness_error"]) < 0.02 * max(span, 1e-6)


def test_attribution_is_not_uniform_across_tokens():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)

    scores = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=64)["scores"]

    assert scores.std() > 1e-8
