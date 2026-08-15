"""Unmerged-adapter loader: attaches a LoRA adapter without merging it into the base weights, so
individual LoRA modules stay addressable for causal ablation."""
import contextlib
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


@contextlib.contextmanager
def toggle(model, active_layers: set[int]):
    """Context manager: for the duration of the block, layers outside `active_layers` are forced
    base-only by zeroing their LoRA modules' `.scaling` (the trained alpha/r multiplier on the
    B(Ax) branch); layers in `active_layers` keep their trained scaling. Restores the original
    scaling on exit, regardless of exceptions inside the block. Not nesting-safe: an outer
    toggle's zeroing can't be undone by an inner toggle's active set."""
    layers = lora_layers(model)
    unknown = active_layers - layers.keys()
    assert not unknown, f"active_layers {unknown} not in model's layers {set(layers)}"
    saved = []  # (module, adapter_name, original_scale)
    for idx, modules in layers.items():
        if idx in active_layers:
            continue
        for module in modules:
            for name in module.scaling:
                saved.append((module, name, module.scaling[name]))
                module.scaling[name] = 0.0
    try:
        yield model
    finally:
        for module, name, scale in saved:
            module.scaling[name] = scale
