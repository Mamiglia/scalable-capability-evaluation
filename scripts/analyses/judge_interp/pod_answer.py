"""Steered answering: the judge model solves benchmark items under steering.

No source text in context, so there is no copy anchor and steering acts
on every token. For each item and condition the model answers the GPQA
prompt with alpha * direction added at the probe layer, then the answer
is scored three ways: correctness against ground truth, projection
along every packed direction (read back unsteered, the same way the
directions were fit), and naturalness (mean log-prob of the produced
tokens under the unsteered model).

Output: answers.jsonl, one record per generation. Judging the texts
head to head is a separate round.

Usage (pod):
  python pod_answer.py --pack interp_pack.npz --items items.parquet \
      --model-dir models/Qwen3-32B --out answers.jsonl \
      --conditions '[{"axis":"none","alpha":0},{"axis":"v_pref_judge","alpha":0.5}]' \
      --seeds 2 --temperature 0.7 [--shard 0/4]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def chat(tokenizer, content):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **kwargs)


def answer_of(text):
    m = re.findall(r"ANSWER:\s*\**\s*([A-D])", text)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", required=True, help="json list of {axis, alpha}")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
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

    items = pd.read_parquet(args.items).iloc[shard_i::shard_n].reset_index(drop=True)

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

    def generate(prompt, axis, alpha, seed):
        steer["vec"] = (None if axis == "none" or alpha == 0
                        else raw_unit(axis) * (alpha * median_norm))
        rendered = chat(tokenizer, prompt)
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(enc["input_ids"].to(model.device),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature or None)
        steer["vec"] = None
        gen = out[0, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        # Naturalness: mean log-prob of what was produced, judged by the
        # unsteered model - a steered-in degradation shows up here.
        with torch.no_grad():
            full = out[:, :].to(model.device)
            logits = model(input_ids=full, use_cache=False).logits[0]
        start = enc["input_ids"].shape[1]
        lp = torch.log_softmax(logits[start - 1:-1].float(), dim=-1)
        tok = full[0, start:]
        nat = float(lp[torch.arange(len(tok)), tok].mean()) if len(tok) else 0.0
        return text, nat

    def read_scores(question, options, transcript):
        rendered = chat(tokenizer, read_template.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["last"] - mean) / std
        return {n: round(float(z @ score_dirs[k]), 4) for k, n in enumerate(names)}

    options_block = "Multiple choice options as shown to the model:\n{options}\n\n"
    total = len(items) * len(conditions) * args.seeds
    k = 0
    with open(args.out, "a") as f:
        for row in items.itertuples():
            for cond in conditions:
                for s in range(args.seeds):
                    k += 1
                    digest = hashlib.sha1(
                        f"{row.sample_id}|{json.dumps(cond, sort_keys=True)}|{s}".encode()).hexdigest()
                    text, nat = generate(row.prompt, cond["axis"], cond["alpha"],
                                         int(digest[:8], 16))
                    given = answer_of(text)
                    scores = read_scores(row.prompt, options_block.format(
                        options=row.options), text)
                    f.write(json.dumps(dict(
                        sample_id=row.sample_id, axis=cond["axis"],
                        alpha=cond["alpha"], seed=s, temperature=args.temperature,
                        target=row.target, given=given,
                        correct=(given == row.target), parsed=given is not None,
                        naturalness=round(nat, 4), n_chars=len(text),
                        reread_scores=scores, answer=text)) + "\n")
                    f.flush()
                    print(f"{k}/{total} {row.sample_id} {cond['axis']} "
                          f"a={cond['alpha']} s={s} given={given} "
                          f"ok={given == row.target} nat={nat:.3f}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
