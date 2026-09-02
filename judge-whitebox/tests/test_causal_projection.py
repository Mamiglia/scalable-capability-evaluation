import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

import causal_projection


def make_tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=4, max_position_embeddings=32)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


class DigitTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [{"1": 1, "2": 2}[text]]


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_projection_hook_removes_the_component():
    torch.manual_seed(1)
    h = torch.randn(3, 5, 8)
    u = torch.stack([torch.from_numpy(unit(np.random.default_rng(s).normal(size=8)))
                     for s in range(3)])
    dose = torch.tensor([1.0, 0.5, 0.0])
    hook = causal_projection._projection_hook(u, dose)
    (out,) = hook(None, None, (h.clone(),))
    before = torch.einsum("bsd,bd->bs", h, u)
    after = torch.einsum("bsd,bd->bs", out, u)
    assert torch.allclose(after[0], torch.zeros_like(after[0]), atol=1e-6)
    assert torch.allclose(after[1], 0.5 * before[1], atol=1e-6)
    assert torch.allclose(after[2], before[2])
    off = h - before[:, :, None] * u[:, None, :]
    assert torch.allclose(out - after[:, :, None] * u[:, None, :], off, atol=1e-6)


def test_condition_table_counts_and_determinism():
    directions = {"a": unit([1, 0, 0, 0]), "b": unit([0, 1, 0, 0])}
    rows = causal_projection.condition_table(directions, 14, [0.5, 1.0], 21, 3, seed=7)
    assert rows[0]["name"] == "baseline"
    assert len(rows) == 1 + 2 * 3 + 3
    layers = [r["layer"] for r in rows]
    assert layers.count(21) == 2
    again = causal_projection.condition_table(directions, 14, [0.5, 1.0], 21, 3, seed=7)
    assert all(np.array_equal(a["vector"], b["vector"]) for a, b in zip(rows[1:], again[1:]))
    for r in rows[1:]:
        assert abs(np.linalg.norm(r["vector"]) - 1.0) < 1e-5


def test_run_prompt_baseline_and_batching_invariance():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    rng = np.random.default_rng(0)
    directions = {"x": unit(rng.normal(size=8))}
    conditions = causal_projection.condition_table(directions, 1, [1.0], 2, 2, seed=0)

    res = causal_projection.run_prompt(model, tokenizer, ids, mask, conditions, batch=4)
    assert all(r is not None for r in res)
    assert res[0]["kl_answer"] == 0.0
    assert all(r["kl_answer"] >= -1e-6 for r in res)
    assert any(abs(r["gap"] - res[0]["gap"]) > 1e-6 for r in res[1:])

    res1 = causal_projection.run_prompt(model, tokenizer, ids, mask, conditions, batch=1)
    for a, b in zip(res, res1):
        assert a["gap"] == pytest.approx(b["gap"], abs=1e-4)
        assert a["nll"] == pytest.approx(b["nll"], abs=1e-4)


def test_zero_dose_matches_baseline():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    u = unit(np.random.default_rng(1).normal(size=8))
    conditions = [{"name": "baseline", "layer": 0, "dose": 0.0, "vector": None},
                  {"name": "x", "layer": 1, "dose": 0.0, "vector": u}]
    res = causal_projection.run_prompt(model, tokenizer, ids, mask, conditions, batch=2)
    assert res[1]["gap"] == pytest.approx(res[0]["gap"], abs=1e-4)
    assert res[1]["kl_answer"] == pytest.approx(0.0, abs=1e-6)


def test_load_directions_round_trip(tmp_path):
    d = {"a": unit([1, 2, 3, 4]), "b": unit([4, 3, 2, 1])}
    np.savez(tmp_path / "dirs.npz", **d)
    back = causal_projection.load_directions(tmp_path / "dirs.npz")
    assert set(back) == {"a", "b"}
    assert np.allclose(back["a"], d["a"])


def fake_meta():
    n = 6
    return {"prompt": "ab cd. ef gh.", "input_ids": list(range(3, 3 + n)),
            "offsets": [(i * 2, i * 2 + 2) for i in range(n)],
            "group_id": np.array([0, 0, 0, 1, 1, 1]),
            "group_names": ["instr_format", "transcript_1"]}


def test_position_mask_variants():
    meta = fake_meta()
    full = causal_projection.position_mask(meta, "full", "x", 6)
    assert full.tolist() == [1.0] * 6
    nl = causal_projection.position_mask(meta, "not_last", "x", 6)
    assert nl.tolist() == [1.0] * 5 + [0.0]
    role = causal_projection.position_mask(meta, "role", "instr_format", 6)
    anti = causal_projection.position_mask(meta, "not_role", "instr_format", 6)
    assert (role + anti).tolist() == [1.0] * 6
    assert role[:3].sum() == 3 and role[3:].sum() == 0
    t1 = causal_projection.role_token_mask(meta, "transcript_1_d5")
    assert t1.tolist() == [False] * 3 + [True] * 3


def test_projection_hook_respects_mask():
    torch.manual_seed(2)
    h = torch.randn(1, 4, 8)
    u = torch.from_numpy(unit(np.random.default_rng(3).normal(size=8)))[None, :].float()
    dose = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    hook = causal_projection._projection_hook(u, dose, mask)
    (out,) = hook(None, None, (h.clone(),))
    assert torch.allclose(out[0, 1], h[0, 1]) and torch.allclose(out[0, 3], h[0, 3])
    proj = torch.einsum("sd,d->s", out[0], u[0])
    assert abs(proj[0]) < 1e-6 and abs(proj[2]) < 1e-6


def test_run_prompt_with_restrictions():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    u = unit(np.random.default_rng(4).normal(size=8))
    conditions = [{"name": "baseline", "layer": 0, "dose": 0.0, "vector": None,
                   "restriction": "full"},
                  {"name": "instr_format", "layer": 1, "dose": 1.0, "vector": u,
                   "restriction": "full"},
                  {"name": "instr_format", "layer": 1, "dose": 1.0, "vector": u,
                   "restriction": "not_last"},
                  {"name": "instr_format", "layer": 1, "dose": 1.0, "vector": u,
                   "restriction": "role"}]
    res = causal_projection.run_prompt(model, tokenizer, ids, mask, conditions,
                                       batch=4, meta=fake_meta())
    assert all(r is not None for r in res)
    gaps = [r["gap"] for r in res]
    assert gaps[1] != gaps[2] or gaps[1] != gaps[3]
