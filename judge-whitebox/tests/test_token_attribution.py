from pathlib import Path

import numpy as np
import pandas as pd
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

    def __call__(self, text, return_tensors=None):
        ids = FAKE_TOKEN_IDS
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids),
                                                                               dtype=torch.long)}

    def decode(self, ids):
        return "".join(str(i) for i in ids)


FAKE_TOKEN_IDS = [3, 4, 5, 6, 7, 8]


def fake_token_meta(row, tokenizer, c):
    """Fixed prompt tokens so no real tokenizer or prompt template is needed."""
    n = len(FAKE_TOKEN_IDS)
    return {"prompt": "abcdef", "input_ids": list(FAKE_TOKEN_IDS),
            "offsets": [(i, i + 1) for i in range(n)],
            "group_id": np.zeros(n, dtype=np.int8),
            "group_names": token_attribution.token_manifold_record.STRUCTURAL_GROUPS}


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


def test_write_attributions_produces_one_row_per_token(tmp_path):
    rows = [{"pair_uid": "p0", "item": "it0", "order": 0, "gap": 1.5,
             "completeness_error": 0.01, "scores": np.array([0.5, 1.0]),
             "input_ids": [3, 4], "offsets": [(0, 1), (1, 2)],
             "group_id": np.array([3, 5], dtype=np.int8),
             "token_text": ["a", "b"]}]

    out = token_attribution.write_attributions(iter(rows), tmp_path / "attr.parquet")

    table = pd.read_parquet(out)
    assert len(table) == 2
    assert table.group.tolist() == ["transcript_1", "answer_readout"]
    assert table.score.tolist() == [0.5, 1.0]


def test_completeness_sweep_reports_one_row_per_prompt_and_step_count(monkeypatch):
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    monkeypatch.setattr(token_attribution.token_manifold_record, "prompt_token_meta",
                        fake_token_meta)
    df = pd.DataFrame([{"pair_uid": "p0", "item": "it0", "order": 0},
                       {"pair_uid": "p1", "item": "it1", "order": 1}])

    table = token_attribution.completeness_sweep(
        model, tokenizer, df, {}, step_counts=(4, 8), n_prompts=2, seed=0)

    assert sorted(table.steps.unique()) == [4, 8]
    assert len(table) == 4
    assert (table.relative_error >= 0).all()


def test_completeness_sweep_error_falls_with_more_steps(monkeypatch):
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    monkeypatch.setattr(token_attribution.token_manifold_record, "prompt_token_meta",
                        fake_token_meta)
    df = pd.DataFrame([{"pair_uid": "p0", "item": "it0", "order": 0}])

    table = token_attribution.completeness_sweep(
        model, tokenizer, df, {}, step_counts=(2, 512), n_prompts=1, seed=0)

    by_steps = table.set_index("steps").relative_error
    assert by_steps[512] < by_steps[2]


def test_attribution_sbatch_exposes_the_sweep():
    text = (Path(__file__).parents[1] / "sbatch" / "token_attribution.sbatch").read_text()

    assert "WB_SWEEP_STEPS" in text
    assert "--sweep-steps" in text


def test_attribution_sbatch_refuses_an_existing_destination():
    text = (Path(__file__).parents[1] / "sbatch" / "token_attribution.sbatch").read_text()

    assert "source ./wb_common.sh" in text
    assert "destination exists" in text
    assert "pip install" not in text
