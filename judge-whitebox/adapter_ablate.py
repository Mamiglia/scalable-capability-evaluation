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
    """decoder layer index -> its LoRA-wrapped linears: modules carrying lora_A whose name ends
    in one of adapter_ablate.target_modules, addressed under a `.layers.N.` path segment."""
    kinds = wbconfig.cfg()["adapter_ablate"]["target_modules"]
    layers = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or name.rsplit(".", 1)[-1] not in kinds:
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not m:
            continue
        layers.setdefault(int(m.group(1)), []).append(module)
    return layers
