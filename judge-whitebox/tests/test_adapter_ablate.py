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


def _toy_peft_model(n_layers=2, d=4, lora_dropout=0.1):
    torch.manual_seed(0)
    base = ToyModel(n_layers, d)
    cfg = peft.LoraConfig(r=1, lora_alpha=1, target_modules=["q_proj"], lora_dropout=lora_dropout)
    return peft.get_peft_model(base, cfg)


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
