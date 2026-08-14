"""Unmerged-adapter loader: attaches a LoRA adapter without merging it into the base weights, so
individual LoRA modules stay addressable for causal ablation."""
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import verdicts
import wbconfig


def load_unmerged(model_name, adapter, c):
    """Mirrors verdicts.load_model, but keeps the adapter attached as an unmerged PeftModel
    (no merge_and_unload) and freezes all parameters."""
    model_path = wbconfig.resolve_model(model_name, c)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)

    device = verdicts.pick_device()
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, device


def lora_layers(model):
    """decoder layer index -> its LoRA-wrapped linears: every module carrying lora_A, addressed
    under a `.layers.N.` path segment. Must be exhaustive over the adapter's LoRA modules, so
    raises rather than skipping a module whose name-tail isn't one of adapter_ablate.target_modules
    or that has no `.layers.N.` segment — either means the adapter reaches somewhere unexpected."""
    kinds = wbconfig.cfg()["adapter_ablate"]["target_modules"]
    layers = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        tail = name.rsplit(".", 1)[-1]
        assert tail in kinds, f"lora_A module {name!r} not in adapter_ablate.target_modules {kinds}"
        m = re.search(r"\.layers\.(\d+)\.", name)
        assert m, f"lora_A module {name!r} has no `.layers.N.` path segment"
        layers.setdefault(int(m.group(1)), []).append(module)
    return layers
