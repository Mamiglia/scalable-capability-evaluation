"""Steered answering by two-stream token mixture.

Pure steering at the strengths that visibly move text (alpha ~2) makes
the model lock into a repetition loop: the steered context makes the
repeated token likelier still, and it ratchets. Mixing fixes that
without weakening the push - each token is drawn from the steered
stream with probability p, else from the unsteered stream, and both
streams then continue from the same emitted prefix, so unsteered
tokens keep pulling the text back onto the manifold.

One batch of two per step (row 0 unsteered, row 1 steered). Records
how often the streams disagreed, which tells whether p had anything to
act on. Answers are scored as in pod_answer.py: correctness against
ground truth, projections along every packed direction, naturalness.

Usage (pod):
  python pod_answer_mix.py --pack interp_pack.npz --items items40.parquet \
      --model-dir models/Qwen3-32B --out mixed.jsonl \
      --conditions '[{"axis":"v_pref_judge","alpha":2.0,"p":0.25}]' \
      --temperature 0.7 [--shard 0/4]
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
    ap.add_argument("--conditions", required=True, help="json list of {axis, alpha, p}")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
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

    steer = {"vec": None}  # row 1 only
    def steer_hook(_m, _i, output):
        if steer["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[0] < 2:
            return output
        h = h.clone()
        h[1] = h[1] + steer["vec"].to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    layer_module = model.model.layers[layer - 1]
    layer_module.register_forward_hook(steer_hook)

    captured = {}
    def read_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    layer_module.register_forward_hook(read_hook)

    eos_ids = {tokenizer.eos_token_id}
    if tokenizer.convert_tokens_to_ids("<|im_end|>") is not None:
        eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    def draw(logits, gen):
        if args.temperature <= 0:
            return int(logits.argmax())
        # sample on cpu: the seeded generator is a cpu one, and mixing
        # device types in multinomial is an error
        p = torch.softmax(logits.float() / args.temperature, dim=-1).cpu()
        return int(torch.multinomial(p, 1, generator=gen))

    def mix_generate(prompt, axis, alpha, p, seed):
        steer["vec"] = (None if axis == "none" or alpha == 0
                        else raw_unit(axis) * (alpha * median_norm))
        rendered = chat(tokenizer, prompt)
        ids = tokenizer(rendered, return_tensors="pt",
                        add_special_tokens=False)["input_ids"]
        batch = ids.repeat(2, 1).to(model.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        past, out_ids, n_disagree, logprob_sum = None, [], 0, 0.0
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                res = model(input_ids=batch if past is None else batch[:, -1:],
                            past_key_values=past, use_cache=True)
                past = res.past_key_values
                lp = torch.log_softmax(res.logits[0, -1].float(), dim=-1)
                unsteered = draw(res.logits[0, -1], gen)
                steered = draw(res.logits[1, -1], gen)
                n_disagree += unsteered != steered
                take_steered = torch.rand(1, generator=gen).item() < p
                token = steered if take_steered else unsteered
                logprob_sum += float(lp[token])
                out_ids.append(token)
                if token in eos_ids:
                    break
                batch = torch.cat([batch, torch.tensor([[token], [token]],
                                                       device=batch.device)], dim=1)
        steer["vec"] = None
        text = tokenizer.decode([t for t in out_ids if t not in eos_ids],
                                skip_special_tokens=True)
        return text, n_disagree, len(out_ids), logprob_sum / max(len(out_ids), 1)

    def read_scores(question, options, transcript):
        rendered = chat(tokenizer, read_template.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["last"] - mean) / std
        return {n: round(float(z @ score_dirs[k]), 4) for k, n in enumerate(names)}

    options_block = "Multiple choice options as shown to the model:\n{options}\n\n"
    total = len(items) * len(conditions)
    k = 0
    with open(args.out, "a") as f:
        for row in items.itertuples():
            for cond in conditions:
                k += 1
                digest = hashlib.sha1(
                    f"{row.sample_id}|{json.dumps(cond, sort_keys=True)}".encode()).hexdigest()
                text, n_dis, n_tok, nat = mix_generate(
                    row.prompt, cond["axis"], cond["alpha"], cond["p"],
                    int(digest[:8], 16))
                given = answer_of(text)
                scores = read_scores(row.prompt, options_block.format(
                    options=row.options), text)
                f.write(json.dumps(dict(
                    sample_id=row.sample_id, axis=cond["axis"], alpha=cond["alpha"],
                    p=cond["p"], temperature=args.temperature, target=row.target,
                    given=given, correct=(given == row.target),
                    parsed=given is not None, n_tokens=n_tok,
                    frac_disagree=round(n_dis / max(n_tok, 1), 4),
                    naturalness=round(nat, 4), n_chars=len(text),
                    reread_scores=scores, answer=text)) + "\n")
                f.flush()
                print(f"{k}/{total} {row.sample_id} {cond['axis']} a={cond['alpha']} "
                      f"p={cond['p']} given={given} ok={given == row.target} "
                      f"dis={n_dis}/{n_tok}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
