"""Steered rewrites along packed directions (runs on the pod).

For each transcript and axis: the judge rewrites the transcript while
alpha * direction is added to the probe layer's residual stream (all
positions, prompt and generation). Alphas are fractions of the median
residual norm, both signs, plus a shared alpha=0 rewrite and a
matched-norm random-direction control. Each rewrite is then re-read
unsteered and scored along every axis. Gates recorded per rewrite:
final answer unchanged, and the re-read score along the steered axis.

Output: rewrites.jsonl, one record per generation.

Usage (pod):
  python pod_steer.py --pack interp_pack.npz --selection selection.parquet \
      --model-dir models/Qwen3-32B --out rewrites.jsonl \
      --n-transcripts 8 --alpha-fracs 0.02,0.05,0.1 \
      --axes approval,v_cap,v_pref,v_pref_orth --random-axis random_a \
      --max-new-tokens 2048
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REWRITE_PROMPT = """Below is a question and an answer transcript. Rewrite the \
answer transcript. Keep the same solution approach and exactly the same final \
answer line. Do not add commentary about the rewrite.

Question:
---
{question}
---

{options}Answer transcript:
---
{transcript}
---

Rewritten answer transcript:"""

READ_PROMPT_KEY = "prompt"  # single-read template inside the pack


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
    ap.add_argument("--n-transcripts", type=int, required=True)
    ap.add_argument("--alpha-fracs", type=str, required=True)
    ap.add_argument("--axes", type=str, required=True)
    ap.add_argument("--random-axis", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steer-positions", default="all", choices=["all", "generate"],
                    help="add the vector at every position, or only while "
                         "generating (the prompt is read unsteered)")
    ap.add_argument("--extra-packs", default="",
                    help="comma-separated extra pack files; the steered axis "
                         "is added at each pack's layer too, scaled by that "
                         "layer's own median norm")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy; >0 samples at this temperature "
                         "(seeded per generation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = np.load(args.pack, allow_pickle=True)
    layer = int(pack["layer"])
    names = list(pack["names"])
    mean = torch.tensor(pack["mean"], dtype=torch.float32)
    std = torch.tensor(pack["std"], dtype=torch.float32)
    median_norm = float(pack["median_norm"])
    read_template = str(pack[READ_PROMPT_KEY])
    score_dirs = torch.stack([torch.tensor(pack[f"dir_{n}"], dtype=torch.float32)
                              for n in names])
    score_dirs = score_dirs / score_dirs.norm(dim=1, keepdim=True)
    # Steering happens in raw activation space: the raw direction that
    # raises the standardized projection of v is v/std.
    def raw_unit(name):
        v = torch.tensor(pack[f"dir_{name}"], dtype=torch.float32) / std
        return v / v.norm()

    alpha_fracs = [float(a) for a in args.alpha_fracs.split(",")]
    axes = args.axes.split(",")
    selection = pd.read_parquet(args.selection)
    top_bottom = selection[selection["role"].isin(["top", "bottom"])]
    chosen = top_bottom.head(args.n_transcripts)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    # One steering slot per layer; extra packs add the same axis at their
    # layers, scaled by their own median norms.
    extras = []
    for path in [p for p in args.extra_packs.split(",") if p]:
        ep = np.load(path, allow_pickle=True)
        extras.append(dict(layer=int(ep["layer"]),
                           std=torch.tensor(ep["std"], dtype=torch.float32),
                           median_norm=float(ep["median_norm"]), pack=ep))
    steer = {layer: None, **{e["layer"]: None for e in extras}}  # layer -> vec
    def make_steer_hook(l):
        def hook(_m, _i, output):
            if steer[l] is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            # generate mode: decode steps carry one position; the
            # multi-token prompt prefill passes through unsteered.
            if args.steer_positions == "generate" and h.shape[1] > 1:
                return output
            h = h + steer[l].to(h.device, h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook
    layer_module = model.model.layers[layer - 1]
    for l in steer:
        model.model.layers[l - 1].register_forward_hook(make_steer_hook(l))

    def extra_raw_unit(e, name):
        v = torch.tensor(e["pack"][f"dir_{name}"], dtype=torch.float32) / e["std"]
        return v / v.norm()

    captured = {}
    def read_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    read_handle = layer_module.register_forward_hook(read_hook)

    def generate(question, options, transcript, axis, frac, run_index):
        if axis == "none":
            for l in steer:
                steer[l] = None
        else:
            steer[layer] = raw_unit(axis) * (frac * median_norm)
            for e in extras:
                steer[e["layer"]] = extra_raw_unit(e, axis) * (frac * e["median_norm"])
        rendered = chat(tokenizer, REWRITE_PROMPT.format(
            question=question, options=options, transcript=transcript))
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        torch.manual_seed(args.seed + run_index)
        with torch.no_grad():
            out = model.generate(enc["input_ids"].to(model.device),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=args.temperature or None)
        for l in steer:
            steer[l] = None
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

    runs = []
    for row in chosen.itertuples():
        runs.append((row, "none", 0.0))
        mid = alpha_fracs[len(alpha_fracs) // 2]
        runs.extend((row, args.random_axis, s * mid) for s in (1, -1))
        for axis in axes:
            for frac in alpha_fracs:
                runs.extend((row, axis, s * frac) for s in (1, -1))
    print(f"{len(runs)} generations queued", flush=True)

    with open(args.out, "a") as f:
        for k, (row, axis, frac) in enumerate(runs):
            rewrite = generate(row.question, row.options, row.text, axis, frac, k)
            scores = read_scores(row.question, row.options, rewrite)
            record = dict(
                text_id=row.text_id, item=row.item, role=row.role,
                axis=axis, alpha_frac=frac,
                answer_original=answer_of(row.text),
                answer_rewrite=answer_of(rewrite),
                answer_unchanged=answer_of(row.text) == answer_of(rewrite),
                n_chars=len(rewrite), rewrite=rewrite, reread_scores=scores)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"{k + 1}/{len(runs)} {row.text_id[:8]} {axis} {frac:+.2f} "
                  f"answer_ok={record['answer_unchanged']} "
                  f"chars={len(rewrite)}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
