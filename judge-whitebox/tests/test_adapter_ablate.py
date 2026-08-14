import peft
import torch
import torch.nn as nn

import adapter_ablate


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
