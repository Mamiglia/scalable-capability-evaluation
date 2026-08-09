"""Steered rewrites under a style-mimic anchor, sampled decoding.

The generator rewrites the transcript keeping approach, steps, final
answer and writing style; steering is added at the probe layer over all
positions. Sampling (temperature > 0, several seeds per condition)
avoids the greedy dead zone the exact-replication anchor hit: the same
condition produces different text, so steering effects are read
distributionally rather than as byte diffs. Each rewrite is re-read
unsteered and scored along every packed direction.

Usage (pod):
  python pod_rewrite.py --pack interp_pack.npz --selection selection.parquet \
      --model-dir models/Qwen3-32B --out rewrites.jsonl \
      --conditions '[{"axis":"none","alpha":0},{"axis":"v_pref_judge","alpha":0.5}]' \
      --seeds 3 --temperature 0.7 [--shard 0/4]
"""

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REWRITE_PROMPT = """Below is a question and an answer transcript. Write your own \
version of this solution: the same approach, the same steps, the same final \
answer line, and the same writing style, voice and habits as the original \
author. Word it yourself throughout - do not reproduce the original's \
sentences. Start with its leading speaker tag (for example "[Assistant]: "). \
Output nothing else. Do not include the --- delimiter lines.

Question:
---
{question}
---

{options}Answer transcript:
---
{transcript}
---

Rewritten answer transcript:"""


def chat(tokenizer, content):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **kwargs)


def answer_of(text):
    m = re.findall(r"ANSWER:\s*([A-Z])", text)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", required=True, help="json list of {axis, alpha}")
    ap.add_argument("--seeds", type=int, required=True,
                    help="generations per (transcript, condition)")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    conditions = json.loads(args.conditions)
    shard_i, shard_n = map(int, args.shard.split("/"))

    pack = np.load(args.pack, allow_pickle=True)
    layer = int(pack["layer"])
    names = list(pack["names"])
    mean = torch.tensor(pack["mean"], dtype=torch.float32)
    std = torch.tensor(pack["std"], dtype=torch.float32)
    median_norm = float(pack["median_norm"])
    read_template = str(pack["prompt"])
    score_dirs = torch.stack([torch.tensor(pack[f"dir_{n}"], dtype=torch.float32)
                              for n in names])
    score_dirs = score_dirs / score_dirs.norm(dim=1, keepdim=True)
    def raw_unit(name):
        v = torch.tensor(pack[f"dir_{name}"], dtype=torch.float32) / std
        return v / v.norm()

    selection = pd.read_parquet(args.selection).iloc[shard_i::shard_n].reset_index(drop=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    steer = {"vec": None}
    def steer_hook(_m, _i, output):
        if steer["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + steer["vec"].to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    layer_module = model.model.layers[layer - 1]
    layer_module.register_forward_hook(steer_hook)

    captured = {}
    def read_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    layer_module.register_forward_hook(read_hook)

    def generate(question, options, transcript, axis, alpha, seed):
        steer["vec"] = (None if axis == "none" or alpha == 0
                        else raw_unit(axis) * (alpha * median_norm))
        rendered = chat(tokenizer, REWRITE_PROMPT.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(enc["input_ids"].to(model.device),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature or None)
        steer["vec"] = None
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    def read_scores(question, options, transcript):
        rendered = chat(tokenizer, read_template.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["last"] - mean) / std
        return {n: float(z @ score_dirs[k]) for k, n in enumerate(names)}

    total = len(selection) * len(conditions) * args.seeds
    k = 0
    with open(args.out, "a") as f:
        for row in selection.itertuples():
            originals = read_scores(row.question, row.options, row.text)
            for cond in conditions:
                for s in range(args.seeds):
                    k += 1
                    digest = hashlib.sha1(
                        f"{row.text_id}|{json.dumps(cond, sort_keys=True)}|{s}".encode()).hexdigest()
                    text = generate(row.question, row.options, row.text,
                                    cond["axis"], cond["alpha"], int(digest[:8], 16))
                    scores = read_scores(row.question, row.options, text)
                    sim = difflib.SequenceMatcher(None, row.text, text).quick_ratio()
                    f.write(json.dumps(dict(
                        text_id=row.text_id, item=row.item, role=row.role,
                        axis=cond["axis"], alpha=cond["alpha"], seed=s,
                        temperature=args.temperature,
                        answer_unchanged=answer_of(row.text) == answer_of(text),
                        similarity=round(sim, 4), n_chars=len(text),
                        original_scores=originals, reread_scores=scores,
                        rewrite=text)) + "\n")
                    f.flush()
                    print(f"{k}/{total} {row.text_id[:8]} {cond['axis']} "
                          f"a={cond['alpha']} s={s} sim={sim:.3f}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
