import copy

import pandas as pd
import peft
import pytest
import torch
import torch.nn as nn

import adapter_ablate
import wbconfig


class ToyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)


class ToyModel(nn.Module):
    """Stand-in decoder stack: .layers[i].q_proj — enough for lora_layers' `.layers.N.` path
    match and for peft's target_modules match on "q_proj"."""

    def __init__(self, n_layers=2, d=4):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(d) for _ in range(n_layers)])

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.q_proj(h)
        return h


def _toy_peft_model(n_layers=2, d=4, lora_dropout=0.1):
    torch.manual_seed(0)
    base = ToyModel(n_layers, d)
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=["q_proj"], lora_dropout=lora_dropout)
    return peft.get_peft_model(base, cfg)


def _toy_peft_model_with_base(n_layers=2, d=4, lora_alpha=2):
    """Same construction as `_toy_peft_model`, no dropout (so it's an exact numerical oracle),
    plus the un-wrapped base (deep-copied before peft touches it) for output comparison."""
    torch.manual_seed(0)
    base = ToyModel(n_layers, d)
    base_ref = copy.deepcopy(base)
    cfg = peft.LoraConfig(r=1, lora_alpha=lora_alpha, target_modules=["q_proj"], lora_dropout=0.0)
    model = peft.get_peft_model(base, cfg)
    model.eval()
    return model, base_ref


class ToyLayerFull(nn.Module):
    """One Linear per adapter_ablate.target_modules kind — the real q/k/v/o/gate/up/down shape."""

    def __init__(self, d, kinds):
        super().__init__()
        for kind in kinds:
            setattr(self, kind, nn.Linear(d, d, bias=False))


class ToyModelFull(nn.Module):
    def __init__(self, kinds, n_layers=2, d=4):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayerFull(d, kinds) for _ in range(n_layers)])


class ToyLayerBadKind(nn.Module):
    """A LoRA-targetable Linear whose name isn't in adapter_ablate.target_modules."""

    def __init__(self, d):
        super().__init__()
        self.lm_head = nn.Linear(d, d, bias=False)


class ToyModelBadKind(nn.Module):
    def __init__(self, n_layers=1, d=4):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayerBadKind(d) for _ in range(n_layers)])


class StubTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"


def test_lora_layers_groups_by_decoder_layer_index():
    model = _toy_peft_model(n_layers=2)
    layers = adapter_ablate.lora_layers(model)
    assert set(layers) == {0, 1}
    assert all(len(mods) == 1 for mods in layers.values())
    assert all(hasattr(m, "lora_A") for mods in layers.values() for m in mods)


def test_lora_layers_groups_all_seven_target_kinds():
    """Normal path: an adapter touching every configured kind still groups cleanly, per-layer."""
    kinds = wbconfig.cfg()["adapter_ablate"]["target_modules"]
    torch.manual_seed(0)
    base = ToyModelFull(kinds, n_layers=2)
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=kinds, lora_dropout=0.0)
    model = peft.get_peft_model(base, cfg)

    layers = adapter_ablate.lora_layers(model)
    assert set(layers) == {0, 1}
    assert all(len(mods) == len(kinds) for mods in layers.values())


def test_lora_layers_raises_on_unexpected_name_tail():
    """A lora_A module outside adapter_ablate.target_modules must not be silently dropped."""
    torch.manual_seed(0)
    base = ToyModelBadKind(n_layers=1)
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=["lm_head"], lora_dropout=0.0)
    model = peft.get_peft_model(base, cfg)

    with pytest.raises(AssertionError, match="lm_head"):
        adapter_ablate.lora_layers(model)


def test_load_unmerged_keeps_lora(monkeypatch):
    base = ToyModel(n_layers=2)
    monkeypatch.setattr(adapter_ablate.AutoTokenizer, "from_pretrained", lambda path: StubTokenizer())
    monkeypatch.setattr(adapter_ablate.AutoModelForCausalLM, "from_pretrained",
                        lambda path, dtype=None: base)

    def fake_peft_from_pretrained(model, adapter):
        # real, unmerged PeftModel — production code must never call merge_and_unload on this
        cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=["q_proj"], lora_dropout=0.1)
        return peft.get_peft_model(model, cfg)
    monkeypatch.setattr(adapter_ablate.PeftModel, "from_pretrained", fake_peft_from_pretrained)

    c = {"models": {"hf_ids": {"toy": "dummy/toy"}}}
    model, tokenizer, device = adapter_ablate.load_unmerged("toy", "fake/adapter", c)

    assert any("lora_A" in n for n, _ in model.named_modules())  # unmerged: LoRA modules survive
    layers = adapter_ablate.lora_layers(model)
    assert set(layers) == {0, 1}

    assert not model.training  # .eval() applied
    assert all(not p.requires_grad for p in model.parameters())  # requires_grad_(False) applied
    for mods in layers.values():
        for mod in mods:
            assert not mod.training
            assert not mod.lora_dropout["default"].training  # dropout disabled in eval


def test_toggle_all_active_matches_merged_output():
    model, _ = _toy_peft_model_with_base(n_layers=2)
    x = torch.randn(1, 4)

    with adapter_ablate.toggle(model, active_layers={0, 1}):
        out_active = model.base_model.model(x)

    out_merged = model.merge_and_unload()(x)  # merges the trained delta into base weights
    assert torch.allclose(out_active, out_merged, atol=1e-4)


def test_toggle_no_active_matches_base_output():
    model, base_ref = _toy_peft_model_with_base(n_layers=2)
    x = torch.randn(1, 4)
    originals = {(m, name): v for mods in adapter_ablate.lora_layers(model).values()
                 for m in mods for name, v in m.scaling.items()}

    with adapter_ablate.toggle(model, active_layers=set()):
        assert all(v == 0.0 for mods in adapter_ablate.lora_layers(model).values()
                   for m in mods for v in m.scaling.values())
        out_off = model.base_model.model(x)

    assert torch.allclose(out_off, base_ref(x), atol=1e-4)
    assert all(m.scaling[name] == v for (m, name), v in originals.items())  # restored


def test_toggle_single_layer_active_is_base_plus_that_layers_delta():
    """Layer 0 active, layer 1 forced off: output must equal base-layer-0-plus-its-LoRA-branch
    fed through base-only layer 1 — computed here directly from the trained lora_A/lora_B,
    independent of `toggle`'s own scaling-zero mechanism."""
    model, base_ref = _toy_peft_model_with_base(n_layers=2)
    x = torch.randn(1, 4)

    lora0 = adapter_ablate.lora_layers(model)[0][0]
    lora1 = adapter_ablate.lora_layers(model)[1][0]
    original_scale1 = lora1.scaling["default"]
    delta0 = lora0.scaling["default"] * lora0.lora_B["default"](lora0.lora_A["default"](x))
    h = base_ref.layers[0].q_proj(x) + delta0
    expected = base_ref.layers[1].q_proj(h)

    with adapter_ablate.toggle(model, active_layers={0}):
        assert lora1.scaling["default"] == 0.0
        out = model.base_model.model(x)

    assert torch.allclose(out, expected, atol=1e-4)
    assert lora1.scaling["default"] == original_scale1  # restored


def test_toggle_rejects_unknown_layer_index():
    model, _ = _toy_peft_model_with_base(n_layers=2)
    with pytest.raises(AssertionError, match="5"):
        with adapter_ablate.toggle(model, active_layers={5}):
            pass


def test_toggle_restores_scaling_even_if_block_raises():
    model, _ = _toy_peft_model_with_base(n_layers=2)
    originals = {(m, name): v for mods in adapter_ablate.lora_layers(model).values()
                 for m in mods for name, v in m.scaling.items()}

    with pytest.raises(ValueError):
        with adapter_ablate.toggle(model, active_layers=set()):
            raise ValueError("boom")

    assert all(m.scaling[name] == v for (m, name), v in originals.items())  # restored


def test_metrics_order_even_odd_hand_computed(capsys):
    """Two pair_uids with both orders, plus a third with only order 0 (excluded, printed)."""
    rows = [
        dict(pair_uid="p1", order=0, l1=3.0, l2=1.0, pick=1, label=1),
        dict(pair_uid="p1", order=1, l1=1.0, l2=4.0, pick=2, label=1),
        dict(pair_uid="p2", order=0, l1=0.5, l2=2.5, pick=2, label=2),
        dict(pair_uid="p2", order=1, l1=2.0, l2=1.0, pick=1, label=2),
        dict(pair_uid="p3", order=0, l1=5.0, l2=0.0, pick=1, label=2),
    ]
    df = pd.DataFrame(rows)
    df["gap"] = df["l1"] - df["l2"]

    out = adapter_ablate.metrics(df)

    # hand-computed: p1 gaps (2.0, -3.0), p2 gaps (-2.0, 1.0); p3 excluded (no order 1)
    assert out["order_even"] == pytest.approx(((2.0 + -3.0) / 2 + (-2.0 + 1.0) / 2) / 2)
    assert out["order_odd"] == pytest.approx(((2.0 - -3.0) / 2 + (-2.0 - 1.0) / 2) / 2)
    assert out["acc"] == pytest.approx((df["pick"] == df["label"]).mean())
    assert out["pos_term"] == pytest.approx(2 * (df["pick"] == 1).mean() - 1)
    assert out["pick_weaker_acc"] == pytest.approx((df["pick"] == 3 - df["label"]).mean())
    assert "excluded 1 pair_uid" in capsys.readouterr().out  # p3 excluded, count printed


class ToyCausalModel(nn.Module):
    """HF-causal-LM-shaped wrapper around a peft-wrapped ToyModel: forward(input_ids) ->
    object with .logits, so verdicts_under's batched-forward pattern runs with no real
    model/tokenizer download."""

    def __init__(self, peft_toy, vocab=50, d=4):
        super().__init__()
        self.peft_toy = peft_toy
        self.embed = nn.Embedding(vocab, d)
        self.readout_proj = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        h = self.peft_toy.base_model.model(x)  # matches toggle tests: bypasses PeftModel.forward
        logits = self.readout_proj(h)

        class Out:
            pass
        out = Out()
        out.logits = logits
        return out


class _FakeBatch(dict):
    def to(self, device):
        return self


class ToyVerdictsTokenizer:
    """Minimal render/pad surface verdicts_under needs; "1"/"2" map to ids 11/22, as in
    test_verdicts.py's stub."""

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
        return "PROMPT"

    def __call__(self, text):
        return {"input_ids": [1, 2, 3]}

    def pad(self, encs, return_tensors):
        return _FakeBatch(input_ids=torch.tensor([e["input_ids"] for e in encs]))

    def encode(self, text, add_special_tokens=False):
        return {"1": [11], "2": [22]}[text]


def test_verdicts_under_calls_toggle_with_the_given_active_layers(monkeypatch):
    """A no-op toggle would still leave columns/scaling looking right, so prove the scoping is
    live by spying on adapter_ablate.toggle itself (delegating to the real one, so restore
    behaviour is still exercised) rather than only checking post-call side effects."""
    peft_toy, _ = _toy_peft_model_with_base(n_layers=2)
    model = ToyCausalModel(peft_toy)
    tokenizer = ToyVerdictsTokenizer()
    df = pd.DataFrame([
        dict(pair_uid="p1", order=0, label=1, question="Q", options_1="", options_2="",
             text_1="t1", text_2="t2"),
        dict(pair_uid="p1", order=1, label=1, question="Q", options_1="", options_2="",
             text_1="t1", text_2="t2"),
    ])
    c = {"verdicts": {"max_input_tokens": 100, "batch_size": 2}}
    originals = {(m, name): v for mods in adapter_ablate.lora_layers(model).values()
                 for m in mods for name, v in m.scaling.items()}

    real_toggle = adapter_ablate.toggle
    calls = []

    def spy_toggle(model_arg, active_layers):
        calls.append(active_layers)
        return real_toggle(model_arg, active_layers)
    monkeypatch.setattr(adapter_ablate, "toggle", spy_toggle)

    out = adapter_ablate.verdicts_under(model, tokenizer, df, active_layers={0}, c=c)

    assert calls == [{0}]  # toggle called exactly once, with the active_layers we passed
    assert set(out.columns) == {"pair_uid", "order", "l1", "l2", "gap", "pick", "label"}
    assert len(out) == 2
    assert (out["gap"] == out["l1"] - out["l2"]).all()
    assert (out["pick"] == out.apply(lambda r: 1 if r["l1"] > r["l2"] else 2, axis=1)).all()
    assert all(m.scaling[name] == v for (m, name), v in originals.items())  # toggle restored
