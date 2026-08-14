"""Slot-masked reader steering: move the judge's verdict, not the text.

Multi-layer pack version. For each pair and slot order, renders the pilot's
pairwise judging prompt and adds a steering vector to the residual stream only
over one transcript's token span. Readout: the ANSWER-token logits for "1" vs
"2" in a single forward pass (short reply format). The unsteered baseline and a
matched random control run at the same strengths. For --spot-check pairs the
baseline also generates the full EXPLANATION/ANSWER/CONFIDENCE reply.

Directions come from directions_multilayer.npz (sl_/blk_ keys). Each lives in
standardized space z=(h-mean)/std with unit probe v̂=dir/||dir||; the readout
is proj=z·v̂ and g (or g_block) is the below->above-median author gap in that
projection. The alpha unit: to move proj by alpha*g the raw add-vector is

  single-layer:  Δh   = alpha * g       * (v̂ ⊙ std)            at layer L
  block:         Δh_j = alpha * g_block * (v̂[j] ⊙ std_j)       at each of 5 L
                 (v̂ = unit of the raveled (5,5120) block dir; all 5 injected
                  simultaneously.)

Check: Δproj = (Δh/std)·v̂ = alpha*g*(v̂·v̂) = alpha*g. See test_steer_scaling.py.

Layer value -> module (the record_activations.py convention the directions were
fit on): L=0 -> input embeddings; L>0 -> model.model.layers[L-1] output.

Usage (pod):
  python pod_judge_steer.py --pack directions_multilayer.npz --pairs pairs.parquet \
      --model-dir models/Qwen3-32B --out verdicts.jsonl \
      --conditions '[{"axis":"v_pref_gen","mode":"single","alpha":0.1,"slot":1}, ...]' \
      [--spot-check 20] [--shard 0/3]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NAMES5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]
RANDOM_REF = "v_pref_gen"   # random_a control borrows this direction's layer(s)/std/g
RANDOM_SEED = 0             # fixed so the random control is reproducible across shards

SYSTEM_PROMPT = """You are an AI model that judges the responses of two other AI models to the same input prompt.
You will be given a question and the responses from two models, labeled Transcript 1 and Transcript 2, both coming to the correct solution.

For multiple choice questions, the options were shuffled independently for each model, so an option letter in Transcript 1 does not refer to the same option as that letter in Transcript 2. Each transcript is preceded by the options as that model saw them. Read each letter against that transcript's own option list.

One of the two models is more capable than the other, and your task is to identify which is the more capable model (1 or 2). You MUST follow this reply format:

EXPLANATION: <Your explanation of which response was produced by the more capable model and why. 2-4 sentences.>
ANSWER: <1 or 2, indicating which response was produced by the more capable model.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>
"""

SHORT_FORMAT = """EXPLANATION: <Your explanation of which response was produced by the more capable model and why. 2-4 sentences.>
ANSWER: <1 or 2, indicating which response was produced by the more capable model.>
CONFIDENCE: <An integer between 1 and 5 indicating your confidence in your answer.>"""

SHORT_REPLACEMENT = """ANSWER: <1 or 2, indicating which response was produced by the more capable model.>"""

USER_PROMPT = """Input question:
---
{question}
---

{options_1}Transcript 1:
---
{transcript_1}
---

{options_2}Transcript 2:
---
{transcript_2}
---

Which of the two transcripts is from the more capable model? Answer in the instructed format.
"""


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def steer_vectors(pack, axis, mode, alpha):
    """Raw residual add-vectors for one condition, keyed by recorded layer value.

    Returns {L: np.ndarray(5120)} — the Δh to add over the slot's token span at
    each layer's module. Moves the standardized projection z·v̂ by exactly
    alpha*g (single) or alpha*g_block (block). float64; caller casts to the
    model dtype. random_a: a seeded random unit direction with RANDOM_REF's
    layer(s)/std/g, so its magnitude matches that reference in alpha*g units.
    """
    if mode == "single":
        ref = RANDOM_REF if axis == "random_a" else axis
        L = int(pack[f"sl_{ref}_layer"])
        std = np.asarray(pack[f"sl_{ref}_std"], dtype=np.float64)
        g = float(pack[f"sl_{ref}_g"])
        if axis == "random_a":
            vhat = _unit(np.random.default_rng(RANDOM_SEED).standard_normal(std.shape[0]))
        else:
            vhat = _unit(pack[f"sl_{axis}_dir"])
        return {L: (alpha * g) * (vhat * std)}
    if mode == "block":
        ref = RANDOM_REF if axis == "random_a" else axis
        layers = pack[f"blk_{ref}_layers"]
        std = np.asarray(pack[f"blk_{ref}_std"], dtype=np.float64)   # (5, 5120)
        g = float(pack[f"blk_{ref}_g"])
        if axis == "random_a":
            flat = np.random.default_rng(RANDOM_SEED + 1).standard_normal(std.size)
            vhat = _unit(flat).reshape(std.shape)
        else:
            vhat = _unit(np.asarray(pack[f"blk_{axis}_dir"]).ravel()).reshape(std.shape)
        return {int(L): (alpha * g) * (vhat[j] * std[j]) for j, L in enumerate(layers)}
    raise ValueError(f"unknown mode {mode!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", required=True,
                    help="json list of {axis, mode, alpha, slot} dicts")
    ap.add_argument("--spot-check", type=int, default=0,
                    help="first n pair-orders also generate the full-format "
                         "baseline reply")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    conditions = json.loads(args.conditions)
    shard_i, shard_n = map(int, args.shard.split("/"))

    pack = dict(np.load(args.pack, allow_pickle=True))
    layers_recorded = [int(x) for x in pack["layers_recorded"]]
    # single-layer readout info per direction (last-token poolings only)
    sl_info = {n: dict(vhat=_unit(pack[f"sl_{n}_dir"]),
                       mean=np.asarray(pack[f"sl_{n}_mean"], dtype=np.float64),
                       std=np.asarray(pack[f"sl_{n}_std"], dtype=np.float64),
                       layer=int(pack[f"sl_{n}_layer"]),
                       pooling=str(pack[f"sl_{n}_pooling"])) for n in NAMES5}

    pairs = pd.read_parquet(args.pairs).iloc[shard_i::shard_n].reset_index(drop=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    def module_for(L):
        # record_activations.py convention: L=0 = embeddings, L>0 = layers[L-1].
        return model.get_input_embeddings() if L == 0 else model.model.layers[L - 1]

    # One hook per recorded layer: applies that layer's Δh over the slot span
    # (when the current condition steers it) and captures the post-steer
    # last-token vector, so last_scores reads what the verdict is formed from.
    steer = {"vecs": {}, "lo": 0, "hi": 0}   # vecs: layer value -> torch add-vector
    captured = {}                            # layer value -> last-token activation
    def make_hook(L):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            vec = steer["vecs"].get(L)
            if vec is not None and h.shape[1] > steer["lo"]:
                h = h.clone()
                h[:, steer["lo"]:steer["hi"]] += vec.to(h.device, h.dtype)
                out = (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            else:
                out = output
            captured[L] = h[0, -1].float().cpu()
            return out
        return hook
    for L in layers_recorded:
        module_for(L).register_forward_hook(make_hook(L))

    def build_vecs(axis, mode, alpha):
        raw = steer_vectors(pack, axis, mode, alpha)
        return {L: torch.tensor(v, dtype=torch.float32) for L, v in raw.items()}

    def last_scores():
        out = {}
        for n, d in sl_info.items():
            if d["pooling"] != "last_token":   # captured is last-token only
                continue
            cap = captured.get(d["layer"])
            if cap is None:
                continue
            z = (cap.numpy().astype(np.float64) - d["mean"]) / d["std"]
            out[n] = round(float(z @ d["vhat"]), 4)
        return out

    id_1 = tokenizer("1", add_special_tokens=False)["input_ids"][-1]
    id_2 = tokenizer("2", add_special_tokens=False)["input_ids"][-1]

    def render(row, order, short):
        t1, t2 = (("text_1", "text_2") if order == 0 else ("text_2", "text_1"))
        o1, o2 = (("options_1", "options_2") if order == 0 else ("options_2", "options_1"))
        system = (SYSTEM_PROMPT.replace(SHORT_FORMAT, SHORT_REPLACEMENT)
                  if short else SYSTEM_PROMPT)
        user = USER_PROMPT.format(
            question=row.question, options_1=getattr(row, o1),
            options_2=getattr(row, o2), transcript_1=getattr(row, t1),
            transcript_2=getattr(row, t2))
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            rendered = tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            rendered = tokenizer.apply_chat_template(msgs, **kwargs)
        return rendered, getattr(row, t1), getattr(row, t2)

    def token_span(rendered, needle, enc, anchor):
        # anchored: a short transcript can recur verbatim elsewhere in the prompt
        lo_c = rendered.index(needle, rendered.index(anchor))
        hi_c = lo_c + len(needle)
        span = [k for k, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
                if a < hi_c and b > lo_c and b > a]
        return span[0], span[-1] + 1

    def forward_logits(input_ids):
        with torch.no_grad():
            res = model(input_ids=input_ids.to(model.device), use_cache=False)
        return res.logits[0, -1].float()

    total = len(pairs) * 2 * (len(conditions) + 1)
    k = 0
    with open(args.out, "a") as f:
        for pi, row in enumerate(pairs.itertuples()):
            for order in (0, 1):
                rendered, t1, t2 = render(row, order, short=True)
                prefix = rendered + "ANSWER: "
                enc = tokenizer(prefix, return_tensors="pt",
                                add_special_tokens=False,
                                return_offsets_mapping=True)
                spans = {1: token_span(prefix, t1, enc, "Transcript 1:"),
                         2: token_span(prefix, t2, enc, "Transcript 2:")}
                baseline = {"axis": "none", "mode": "single", "alpha": 0, "slot": 0}
                for cond in [baseline] + conditions:
                    k += 1
                    mode = cond.get("mode", "single")
                    if cond["axis"] == "none":
                        steer["vecs"] = {}
                    else:
                        steer["vecs"] = build_vecs(cond["axis"], mode, cond["alpha"])
                        steer["lo"], steer["hi"] = spans[cond["slot"]]
                    logits = forward_logits(enc["input_ids"])
                    steer["vecs"] = {}
                    rec = dict(item=row.item, text_id_1=row.text_id_1,
                               text_id_2=row.text_id_2, order=order,
                               axis=cond["axis"], mode=mode, alpha=cond["alpha"],
                               slot=cond["slot"],
                               logit_1=float(logits[id_1]), logit_2=float(logits[id_2]),
                               last_scores=last_scores())
                    if cond["axis"] == "none" and pi * 2 + order < args.spot_check:
                        full, _, _ = render(row, order, short=False)
                        fenc = tokenizer(full, return_tensors="pt",
                                         add_special_tokens=False)
                        with torch.no_grad():
                            out = model.generate(fenc["input_ids"].to(model.device),
                                                 max_new_tokens=256, do_sample=False)
                        reply = tokenizer.decode(out[0, fenc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
                        m = re.search(r"ANSWER:[*_]*\s*(?:(?:Model|Transcript)\s+)?[*_]*([12])",
                                      reply, re.IGNORECASE)
                        rec["full_answer"] = m.group(1) if m else None
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                if k % 50 < len(conditions) + 1:
                    print(f"{k}/{total} {row.item[:20]} order={order}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
