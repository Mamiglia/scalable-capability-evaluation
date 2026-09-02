"""Per-token direction maps + sentence occlusion, multilayer pack (runs on pod).

Multi-layer version of pod_token_maps.py. Uses the five single-layer
directions from directions_multilayer.npz (sl_<name>_*). Each direction sits
at its own recorded layer (e.g. v_cap/v_pref_gen at L26, v_pref_spec at L50),
so one forward pass captures the UNION of those layers and each direction is
projected at its own layer, restricted to the transcript's token span.

Per selected transcript: per-token projection z·v̂ (z=(h_at_that_layer-mean)/std)
for each direction, the last-token score, and sentence occlusion - delete one
sentence, re-read, record the change in each direction's last-token score plus
the sentence length. One json per (text_id, item). Feeds the interactive
sentence dashboard (which sentences drive each axis up vs down).

Layer -> module (record_activations.py convention the directions were fit on):
L=0 = input embeddings; L>0 = model.model.layers[L-1] output.

Self-contained: needs directions_multilayer.npz, selection.parquet, weights.

Usage (pod):
  python pod_token_maps_ml.py --pack directions_multilayer.npz \
      --selection selection.parquet --model-dir models/Qwen3-32B --out token_maps

Self-test (CPU, no model):
  python pod_token_maps_ml.py --self-test --pack directions_multilayer.npz
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAMES5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def load_pack(path):
    """Read the five single-layer directions and the fixed prompt template.

    Returns (info, template). info: per-name vhat/mean/std/layer (all float64),
    plus 'names' and 'needed_layers' (sorted union of the directions' layers).
    """
    pack = np.load(path, allow_pickle=True)
    info = {"names": NAMES5, "vhat": {}, "mean": {}, "std": {}, "layer": {}}
    for n in NAMES5:
        info["vhat"][n] = _unit(pack[f"sl_{n}_dir"])
        info["mean"][n] = np.asarray(pack[f"sl_{n}_mean"], dtype=np.float64)
        info["std"][n] = np.asarray(pack[f"sl_{n}_std"], dtype=np.float64)
        info["layer"][n] = int(pack[f"sl_{n}_layer"])
    info["needed_layers"] = sorted({info["layer"][n] for n in NAMES5})
    return info, str(pack["prompt"])


def project(captured, info):
    """Per-token projections, one column per direction (tokens, 5).

    captured: {layer value -> (tokens, 5120) hidden states}. Each direction is
    projected at its own layer: z=(h-mean)/std, proj=z·v̂.
    """
    cols = []
    for n in info["names"]:
        h = np.asarray(captured[info["layer"][n]], dtype=np.float64)
        z = (h - info["mean"][n]) / info["std"][n]
        cols.append(z @ info["vhat"][n])
    return np.stack(cols, axis=1)


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


def self_test(pack_path):
    """CPU-only: needed-layers set, projection shapes, mean-vector ~0."""
    info, template = load_pack(pack_path)
    assert info["needed_layers"] == [26, 50], info["needed_layers"]
    assert "{transcript}" in template and "{question}" in template
    print(f"needed layers {info['needed_layers']} (== [26, 50]): OK")

    rng = np.random.default_rng(0)
    n_tok, dim = 7, info["mean"][NAMES5[0]].shape[0]
    captured = {L: rng.standard_normal((n_tok, dim)) for L in info["needed_layers"]}
    proj = project(captured, info)
    assert proj.shape == (n_tok, len(NAMES5)), proj.shape
    print(f"random hidden {n_tok}x{dim} -> proj {proj.shape}: OK")

    # projecting each direction's own mean vector gives ~0 at its own layer
    for n in NAMES5:
        mean_cap = {L: np.tile(info["mean"][n], (n_tok, 1))
                    for L in info["needed_layers"]}
        z = project(mean_cap, info)[:, NAMES5.index(n)]
        assert np.abs(z).max() < 1e-6, (n, float(np.abs(z).max()))
    print("mean-vector projection ~0 for every direction (<1e-6): OK")
    print("self-test PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU-only checks against the real pack; no model")
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th transcript")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.pack)
        return
    for req in ("selection", "model_dir", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")
    args.out.mkdir(parents=True, exist_ok=True)

    info, template = load_pack(args.pack)
    shard_i, shard_n = map(int, args.shard.split("/"))
    selection = pd.read_parquet(args.selection).iloc[shard_i::shard_n]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), dtype=getattr(torch, args.dtype),
        device_map=args.device_map)
    model.eval()

    def module_for(L):
        return model.get_input_embeddings() if L == 0 else model.model.layers[L - 1]

    captured = {}   # layer value -> (tokens, 5120) numpy

    def make_hook(L):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[L] = h[0].float().cpu().numpy()
        return hook
    for L in info["needed_layers"]:
        module_for(L).register_forward_hook(make_hook(L))

    def read(question, options, transcript):
        """(per-token projections (tokens,5), tokens, transcript token span).

        The span matters: the prompt's own instruction text shares the sequence,
        and without it the top-projecting tokens are prompt words, not the
        author's. Directions were fit at the last position of this prompt.
        """
        rendered = build_prompt(tokenizer, template, question, options, transcript)
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True)
        with torch.no_grad():
            model.model(input_ids=enc["input_ids"].to(model.device), use_cache=False)
        proj = project(captured, info)  # (tokens, 5)
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
                            for k, n in enumerate(NAMES5)}))
        result = dict(
            text_id=row.text_id, item=row.item,
            role=getattr(row, "role", ""), elo=float(getattr(row, "elo", 0.0)),
            directions=NAMES5, tokens=tokens, transcript_span=list(span),
            token_projections={n: proj[:, k].round(3).tolist()
                               for k, n in enumerate(NAMES5)},
            last_token_scores={n: float(proj[-1, k]) for k, n in enumerate(NAMES5)},
            occlusion=occlusion)
        out_path.write_text(json.dumps(result))
        print(f"{row.text_id}_{row.item}: {len(tokens)} tokens, "
              f"{len(sentences)} sentences", flush=True)
    print(f"-> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
