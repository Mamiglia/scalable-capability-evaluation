import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    rows = [{"pair_uid": "p0", "item": "it0", "order": 0, "gap": 1.5, "baseline_gap": 0.25,
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


def test_batched_steps_match_one_step_at_a_time():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids)

    one = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=9, step_batch=1)
    many = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=9, step_batch=4)

    assert np.allclose(one["scores"], many["scores"], rtol=0, atol=1e-6)
    assert abs(one["gap"] - many["gap"]) < 1e-6


def test_shards_partition_the_prompts():
    df = pd.DataFrame({"pair_uid": [f"p{i // 2}" for i in range(10)],
                       "order": [i % 2 for i in range(10)]})

    shards = [token_attribution.select_shard(df, i, 3) for i in range(3)]

    assert [len(s) for s in shards] == [4, 3, 3]
    seen = pd.concat(shards).sort_index()
    assert seen.index.is_unique
    assert seen.equals(df)


def test_parser_accepts_both_folds_and_all():
    parser = token_attribution.build_parser({"verdicts": {"eval_split": "test"}})

    for fold in ("A", "B", "all"):
        args = parser.parse_args(["--model", "m", "--fold", fold, "--out", "o.parquet"])
        assert args.fold == fold


def test_parser_defaults_a_single_shard_and_a_step_batch():
    parser = token_attribution.build_parser({"verdicts": {"eval_split": "test"}})

    args = parser.parse_args(["--model", "m", "--fold", "A", "--out", "o.parquet"])

    assert (args.shard_index, args.shard_count) == (0, 1)
    assert args.step_batch == token_attribution.STEP_BATCH_DEFAULT


def test_attribution_sbatch_runs_as_a_shard_array():
    text = (Path(__file__).parents[1] / "sbatch" / "token_attribution.sbatch").read_text()

    assert "SLURM_ARRAY_TASK_ID" in text
    assert "WB_SHARDS" in text
    assert "--shard-index" in text and "--shard-count" in text
    dests = [line for line in text.splitlines() if line.strip().startswith("DEST=")]
    assert dests and all("${SHARD_SUFFIX}" in line for line in dests)


def test_attribution_sbatch_exposes_the_sweep():
    text = (Path(__file__).parents[1] / "sbatch" / "token_attribution.sbatch").read_text()

    assert "WB_SWEEP_STEPS" in text
    assert "--sweep-steps" in text


def test_attribution_sbatch_refuses_an_existing_destination():
    text = (Path(__file__).parents[1] / "sbatch" / "token_attribution.sbatch").read_text()

    assert "source ./wb_common.sh" in text
    assert "destination exists" in text
    assert "pip install" not in text


def test_attribution_does_not_populate_parameter_gradients():
    """Only the input embeddings need gradients; weight grads would waste the GPU."""
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6]])

    token_attribution.attribute_tokens(model, tokenizer, ids, torch.ones_like(ids), steps=4)

    assert all(p.grad is None for p in model.parameters())
    assert not any(p.requires_grad for p in model.parameters())


def test_attribution_table_keeps_the_baseline_gap_and_character_offsets(tmp_path):
    """The baseline offset and offsets are needed downstream; dropping them means re-running."""
    rows = [{"pair_uid": "p0", "item": "it0", "order": 0, "gap": 1.5, "baseline_gap": 0.25,
             "completeness_error": 0.01, "scores": np.array([0.5, 1.0]),
             "input_ids": [3, 4], "offsets": [(0, 1), (1, 5)],
             "group_id": np.array([3, 5], dtype=np.int8), "token_text": ["a", "bcde"]}]

    out = token_attribution.write_attributions(iter(rows), tmp_path / "a.parquet")

    table = pd.read_parquet(out)
    assert table.baseline_gap.tolist() == [0.25, 0.25]
    assert table.char_start.tolist() == [0, 1]
    assert table.char_end.tolist() == [1, 5]


def test_gap_shortcut_matches_running_the_full_head():
    """Projecting the last hidden state must equal the answer-logit difference."""
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    mask = torch.ones_like(ids)
    id1, id2 = token_attribution.answer_token_ids(tokenizer)

    with torch.no_grad():
        shortcut = token_attribution._gap_logits(model, model.get_input_embeddings()(ids),
                                                 mask, id1, id2)
        full = model(input_ids=ids).logits[:, -1]

    assert shortcut.item() == pytest.approx((full[0, id1] - full[0, id2]).item(), abs=1e-5)


def test_attribution_refuses_when_the_last_position_is_padding():
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])

    with pytest.raises(AssertionError, match="padding"):
        token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=2)


def test_attribution_defaults_to_the_precision_the_judges_run_at():
    """bfloat16 matches verdicts.load_model, so attribution explains the deployed model."""
    import verdicts

    parser = token_attribution.build_parser({"verdicts": {"eval_split": "eval_mm"}})
    args = parser.parse_args(["--model", "m", "--fold", "all", "--out", "o"])

    assert args.dtype == "bfloat16"
    assert inspect.signature(verdicts.load_model).parameters["dtype"].default == torch.bfloat16


def _fake_row(uid, order):
    return {"pair_uid": uid, "item": "it0", "order": order, "gap": 1.0, "baseline_gap": 0.1,
            "completeness_error": 0.0, "scores": np.array([0.5, 0.5]), "input_ids": [3, 4],
            "offsets": [(0, 1), (1, 2)], "group_id": np.array([3, 5], dtype=np.int8),
            "token_text": ["a", "b"]}


def test_parts_are_written_as_the_run_goes_and_resume_skips_them(tmp_path):
    """A long metered run must not lose everything to one failure."""
    rows = [_fake_row(f"p{i}", i % 2) for i in range(5)]

    written = token_attribution.write_attribution_parts(iter(rows), tmp_path, chunk_prompts=2)

    assert len(written) == 3
    assert token_attribution.completed_prompts(tmp_path) == {
        ("p0", 0), ("p1", 1), ("p2", 0), ("p3", 1), ("p4", 0)}
    assert len(token_attribution.read_attribution_parts(tmp_path)) == 10


def test_resuming_appends_parts_without_overwriting(tmp_path):
    token_attribution.write_attribution_parts(iter([_fake_row("p0", 0)]), tmp_path, 1)

    token_attribution.write_attribution_parts(iter([_fake_row("p1", 1)]), tmp_path, 1)

    assert sorted(p.name for p in tmp_path.glob("part_*.parquet")) == [
        "part_00000.parquet", "part_00001.parquet"]
    assert token_attribution.completed_prompts(tmp_path) == {("p0", 0), ("p1", 1)}


def test_mean_baseline_is_not_degenerate():
    """A zero baseline gives a zero hidden state under RMSNorm, so the path is pathological."""
    model, tokenizer = make_tiny_model(), DigitTokenizer()
    ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.ones_like(ids)

    zeros = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=8,
                                               baseline_kind="zeros")
    mean = token_attribution.attribute_tokens(model, tokenizer, ids, mask, steps=8,
                                              baseline_kind="mean")

    assert zeros["baseline_gap"] == pytest.approx(0.0, abs=1e-6)
    assert mean["baseline_gap"] != pytest.approx(0.0, abs=1e-6)
