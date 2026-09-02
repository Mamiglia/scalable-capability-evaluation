"""Per-token direction maps + sentence occlusion (runs on the pod).

For each selected transcript: one forward pass recording every token
position at the probe layer, projected onto every packed direction -
the token-level heatmap of what sits high or low along each axis.
Then sentence occlusion: delete one sentence at a time, re-read, and
record the change in the last-token scores. Output: one json per
transcript with tokens, per-direction token projections, and the
occlusion table. Self-contained: needs interp_pack.npz,
selection.parquet, and the model weights.

Usage (pod):
  python pod_token_maps.py --pack interp_pack.npz --selection selection.parquet \
      --model-dir models/Qwen3-32B --out token_maps
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


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th transcript")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pack = np.load(args.pack, allow_pickle=True)
    layer = int(pack["layer"])
    names = list(pack["names"])
    mean = torch.tensor(pack["mean"], dtype=torch.float32)
    std = torch.tensor(pack["std"], dtype=torch.float32)
    dirs = torch.stack([torch.tensor(pack[f"dir_{n}"], dtype=torch.float32)
                        for n in names])
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    template = str(pack["prompt"])
    shard_i, shard_n = map(int, args.shard.split("/"))
    selection = pd.read_parquet(args.selection).iloc[shard_i::shard_n]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    captured = {}
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0].float().cpu()
    model.model.layers[layer - 1].register_forward_hook(hook)

    def read(question, options, transcript):
        """(per-token z-projections, tokens, transcript token span).

        The span matters: the prompt's own instruction text is in the
        same sequence, and without it the highest-projecting tokens are
        prompt words rather than anything the author wrote.
        """
        rendered = build_prompt(tokenizer, template, question, options, transcript)
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        z = (captured["h"] - mean) / std
        proj = (z @ dirs.T).numpy()  # (tokens, dirs)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        lo_c = rendered.rfind(transcript[:60])
        hi_c = lo_c + len(transcript)
        span = [k for k, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
                if a < hi_c and b > lo_c and b > a]
        return proj, tokens, ((span[0], span[-1] + 1) if span else (0, len(tokens)))

    for row in selection.itertuples():
        out_path = args.out / f"{row.text_id}_{row.item}.json"
        if out_path.exists():          # resume: rereading is the expensive part
            continue
        proj, tokens, span = read(row.question, row.options, row.text)
        occlusion = []
        sentences = split_sentences(row.text)
        for si in range(len(sentences)):
            reduced = " ".join(sentences[:si] + sentences[si + 1:])
            proj_r, _, _ = read(row.question, row.options, reduced)
            occlusion.append(dict(
                sentence=sentences[si], n_chars=len(sentences[si]),
                delta_last={n: float(proj[-1, k] - proj_r[-1, k])
                            for k, n in enumerate(names)}))
        result = dict(
            text_id=row.text_id, item=row.item,
            role=getattr(row, "role", ""), elo=float(getattr(row, "elo", 0.0)),
            directions=names, tokens=tokens, transcript_span=list(span),
            token_projections={n: proj[:, k].round(3).tolist()
                               for k, n in enumerate(names)},
            last_token_scores={n: float(proj[-1, k]) for k, n in enumerate(names)},
            occlusion=occlusion)
        out_path.write_text(json.dumps(result))
        print(f"{row.text_id}_{row.item}: {len(tokens)} tokens, "
              f"{len(sentences)} sentences", flush=True)
    print(f"-> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
