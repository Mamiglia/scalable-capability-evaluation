"""Token-mixture steering under an exact-replication anchor (pod).

Both streams get the same instruction: reproduce the transcript
exactly. At each generated token, the steered stream's greedy token is
emitted with probability p, the unsteered stream's with 1-p; both
streams then continue from the same mixed prefix (one batch-of-2
forward per step). p=0 is pure unsteered replication (approximately
the original), p=1 fully steered replication. Each blend is re-read
unsteered and scored along every packed direction; the fraction of
tokens where the two streams disagreed and the char-level similarity
to the original are recorded.

Usage (pod):
  python pod_mix.py --pack interp_pack.npz --selection selection.parquet \
      --model-dir models/Qwen3-32B --out mixes.jsonl \
      --conditions '[{"axis":"none","alpha":0,"p":0},{"axis":"approval","alpha":0.4,"p":0.5}]' \
      --seed 0 [--shard 0/4] [--device-map auto] [--dtype bfloat16]
"""

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPLICATE_PROMPT = """Below is a question and an answer transcript. Reproduce \
the answer transcript exactly, word for word, starting with its leading \
speaker tag (for example "[Assistant]: "). Output nothing else. Do not \
include the --- delimiter lines.

Question:
---
{question}
---

{options}Answer transcript:
---
{transcript}
---

Exact reproduction:"""


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
    ap.add_argument("--conditions", required=True,
                    help="json list of {axis, alpha, p} dicts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nat-floor", type=float, default=None,
                    help="mean log-prob per emitted token below which a blend "
                         "is retried once with a fresh seed (calibrate from "
                         "the p=0 baseline round)")
    ap.add_argument("--extra-packs", default="",
                    help="comma-separated extra pack files; the steered axis "
                         "is added at each pack's layer too, scaled by that "
                         "layer's own median norm")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th transcript")
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

    selection = pd.read_parquet(args.selection)
    selection = selection.iloc[shard_i::shard_n].reset_index(drop=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    extras = []
    for path in [p for p in args.extra_packs.split(",") if p]:
        ep = np.load(path, allow_pickle=True)
        extras.append(dict(layer=int(ep["layer"]),
                           std=torch.tensor(ep["std"], dtype=torch.float32),
                           median_norm=float(ep["median_norm"]), pack=ep))
    def extra_raw_unit(e, name):
        v = torch.tensor(e["pack"][f"dir_{name}"], dtype=torch.float32) / e["std"]
        return v / v.norm()

    steer = {layer: None, **{e["layer"]: None for e in extras}}  # layer -> vec, row 1 only
    def make_steer_hook(l):
        def hook(_m, _i, output):
            if steer[l] is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[0] < 2:
                return output
            h = h.clone()
            h[1] = h[1] + steer[l].to(h.device, h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook
    for l in steer:
        model.model.layers[l - 1].register_forward_hook(make_steer_hook(l))

    captured = {}
    def read_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    model.model.layers[layer - 1].register_forward_hook(read_hook)

    def mix_generate(question, options, transcript, axis, alpha, p, rng):
        if axis == "none" or alpha == 0:
            for l in steer:
                steer[l] = None
        else:
            steer[layer] = raw_unit(axis) * (alpha * median_norm)
            for e in extras:
                steer[e["layer"]] = extra_raw_unit(e, axis) * (alpha * e["median_norm"])
        rendered = chat(tokenizer, REPLICATE_PROMPT.format(
            question=question, options=options, transcript=transcript))
        ids = tokenizer(rendered, return_tensors="pt",
                        add_special_tokens=False)["input_ids"]
        max_new = int(len(tokenizer(transcript)["input_ids"]) * 1.15) + 48
        batch = ids.repeat(2, 1).to(model.device)
        past = None
        out_ids = []
        n_disagree = 0
        logprob_sum = 0.0  # emitted tokens under the unsteered stream:
        eos = tokenizer.eos_token_id  # the naturalness gate, free per step
        with torch.no_grad():
            for step in range(max_new):
                res = model(input_ids=batch if past is None else batch[:, -1:],
                            past_key_values=past, use_cache=True)
                past = res.past_key_values
                logp = torch.log_softmax(res.logits[0, -1].float(), dim=-1)
                unsteered = int(res.logits[0, -1].argmax())
                steered = int(res.logits[1, -1].argmax())
                n_disagree += unsteered != steered
                token = steered if rng.random() < p else unsteered
                logprob_sum += float(logp[token])
                out_ids.append(token)
                if token == eos:
                    break
                batch = torch.cat([batch, torch.tensor([[token], [token]],
                                                       device=batch.device)], dim=1)
        for l in steer:
            steer[l] = None
        text = tokenizer.decode([t for t in out_ids if t != eos],
                                skip_special_tokens=True)
        nat = logprob_sum / max(len(out_ids), 1)  # mean log-prob per token
        return text, n_disagree, len(out_ids), nat

    def read_scores(question, options, transcript):
        rendered = chat(tokenizer, read_template.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["last"] - mean) / std
        return {n: float(z @ score_dirs[k]) for k, n in enumerate(names)}

    total = len(selection) * len(conditions)
    k = 0
    with open(args.out, "a") as f:
        for row in selection.itertuples():
            for cond in conditions:
                k += 1
                # hashlib, not hash(): python randomizes str hashes per
                # process, which would unseed the mixture across pods.
                import hashlib
                digest = hashlib.sha1(
                    f"{row.text_id}|{json.dumps(cond, sort_keys=True)}".encode()).hexdigest()
                base_seed = args.seed + int(digest[:8], 16)
                retried = False
                for attempt in (0, 1):
                    rng = np.random.default_rng(base_seed + attempt)
                    text, n_dis, n_tok, nat = mix_generate(
                        row.question, row.options, row.text,
                        cond["axis"], cond["alpha"], cond["p"], rng)
                    if args.nat_floor is None or nat >= args.nat_floor:
                        break
                    retried = True
                scores = read_scores(row.question, row.options, text)
                sim = difflib.SequenceMatcher(None, row.text, text).quick_ratio()
                f.write(json.dumps(dict(
                    text_id=row.text_id, item=row.item, role=row.role,
                    axis=cond["axis"], alpha=cond["alpha"], p=cond["p"],
                    answer_unchanged=answer_of(row.text) == answer_of(text),
                    similarity=round(sim, 4), n_tokens=n_tok,
                    frac_disagree=round(n_dis / max(n_tok, 1), 4),
                    naturalness=round(nat, 4), retried=retried,
                    reread_scores=scores, blend=text)) + "\n")
                f.flush()
                print(f"{k}/{total} {row.text_id[:8]} {cond['axis']} "
                      f"a={cond['alpha']} p={cond['p']} sim={sim:.3f} "
                      f"dis={n_dis}/{n_tok} nat={nat:.2f}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
