"""Slot-masked reader steering: move the judge's verdict, not the text.

For each pair and slot order, renders the pilot's pairwise judging
prompt and adds alpha * direction to the residual stream only over one
transcript's token span. Readout: the ANSWER-token logits for "1" vs
"2" in a single forward pass (short reply format). A matched random
direction and the unsteered baseline run at the same strengths. For
--spot-check pairs the baseline also generates the pilot's full
EXPLANATION/ANSWER/CONFIDENCE reply, to confirm the short format
tracks real verdicts.

Usage (pod):
  python pod_judge_steer.py --pack interp_pack.npz --pairs pairs.parquet \
      --model-dir models/Qwen3-32B --out verdicts.jsonl \
      --conditions '[{"axis":"approval","alpha":0.1,"slot":1}, ...]' \
      [--spot-check 20] [--shard 0/3]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", required=True,
                    help="json list of {axis, alpha, slot} dicts")
    ap.add_argument("--spot-check", type=int, default=0,
                    help="first n pair-orders also generate the full-format "
                         "baseline reply")
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
    score_dirs = torch.stack([torch.tensor(pack[f"dir_{n}"], dtype=torch.float32)
                              for n in names])
    score_dirs = score_dirs / score_dirs.norm(dim=1, keepdim=True)
    def raw_unit(name):
        v = torch.tensor(pack[f"dir_{name}"], dtype=torch.float32) / std
        return v / v.norm()

    pairs = pd.read_parquet(args.pairs).iloc[shard_i::shard_n].reset_index(drop=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    steer = {"vec": None, "lo": 0, "hi": 0}
    def steer_hook(_m, _i, output):
        if steer["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] <= steer["lo"]:  # decode steps past the span
            return output
        h = h.clone()
        h[:, steer["lo"]:steer["hi"]] += steer["vec"].to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    layer_module = model.model.layers[layer - 1]
    layer_module.register_forward_hook(steer_hook)

    # Read the axis values the verdict is actually formed from: the same
    # position the directions were fit at, under whatever steering is on.
    captured = {}
    def read_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[0, -1].float().cpu()
    layer_module.register_forward_hook(read_hook)

    def last_scores():
        z = (captured["last"] - mean) / std
        return {n: round(float(z @ score_dirs[k]), 4) for k, n in enumerate(names)}

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
                for cond in [{"axis": "none", "alpha": 0, "slot": 0}] + conditions:
                    k += 1
                    if cond["axis"] == "none":
                        steer["vec"] = None
                    else:
                        steer["vec"] = raw_unit(cond["axis"]) * (cond["alpha"] * median_norm)
                        steer["lo"], steer["hi"] = spans[cond["slot"]]
                    logits = forward_logits(enc["input_ids"])
                    steer["vec"] = None
                    rec = dict(item=row.item, text_id_1=row.text_id_1,
                               text_id_2=row.text_id_2, order=order,
                               axis=cond["axis"], alpha=cond["alpha"],
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
