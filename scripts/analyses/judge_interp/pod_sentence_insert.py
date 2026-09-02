"""Insertion test: does adding a sentence move the axis in a new transcript?

Occlusion only says a sentence mattered where it already sat. Insertion
asks the causal question in the reading frame: take a donor sentence,
paste it into a transcript that never had it, and re-read. A sentence
that raises the score across many unrelated hosts is carrying the
feature; one that only mattered at home was context.

Each donor is inserted at a fixed relative position in each host (just
before the final answer line if there is one, else at the end), and the
host is also read unchanged, so the measure is a within-host delta.

Usage (pod):
  python pod_sentence_insert.py --pack interp_pack.npz --hosts hosts.parquet \
      --donors donors.json --model-dir models/Qwen3-32B --out inserts.jsonl \
      [--shard 0/4]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_prompt(tokenizer, template, question, options, transcript):
    message = template.format(question=question, options=options,
                              transcript=transcript)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], **kwargs)


def insert_sentence(text: str, sentence: str) -> str:
    """Before the final answer line, so the insertion never displaces it."""
    m = list(re.finditer(r"\n\s*(?:\*\*)?ANSWER:", text))
    if m:
        cut = m[-1].start()
        return text[:cut] + "\n" + sentence.strip() + text[cut:]
    return text.rstrip() + " " + sentence.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))

    pack = np.load(args.pack, allow_pickle=True)
    layer = int(pack["layer"])
    names = list(pack["names"])
    mean = torch.tensor(pack["mean"], dtype=torch.float32)
    std = torch.tensor(pack["std"], dtype=torch.float32)
    dirs = torch.stack([torch.tensor(pack[f"dir_{n}"], dtype=torch.float32)
                        for n in names])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    template = str(pack["prompt"])

    hosts = pd.read_parquet(args.hosts).iloc[shard_i::shard_n].reset_index(drop=True)
    donors = json.loads(args.donors.read_text())   # [{id, category, sentence}]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    captured = {}
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    model.model.layers[layer - 1].register_forward_hook(hook)

    def score(question, options, transcript):
        rendered = build_prompt(tokenizer, template, question, options, transcript)
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["last"] - mean) / std
        return {n: float(z @ dirs[k]) for k, n in enumerate(names)}

    total = len(hosts) * (len(donors) + 1)
    k = 0
    with open(args.out, "a") as f:
        for row in hosts.itertuples():
            base = score(row.question, row.options, row.text)
            k += 1
            for d in donors:
                k += 1
                text = insert_sentence(row.text, d["sentence"])
                s = score(row.question, row.options, text)
                f.write(json.dumps(dict(
                    text_id=row.text_id, item=row.item, donor_id=d["id"],
                    category=d.get("category", ""), n_chars=len(d["sentence"]),
                    delta={n: round(s[n] - base[n], 4) for n in names},
                    base={n: round(base[n], 4) for n in names})) + "\n")
                f.flush()
            print(f"{k}/{total} host {row.text_id[:8]}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
