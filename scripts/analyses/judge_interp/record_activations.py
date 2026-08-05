"""Record judge activations per transcript (spec step 2).

For each (text_id, item) in staging/inputs.parquet, runs the single-
transcript read prompt through the judge and saves, at every
layer_step-th layer: the last-token vector and the mean over the
transcript's own tokens. Output: one float16 .npy of shape
(n_layers_kept, 2, hidden) per entity, in acts_dir. Entities whose file
already exists are skipped, so the job resumes cleanly after
preemption. Nothing on storage is ever deleted or overwritten.

Self-contained: needs only this file, config.yaml, the staged inputs
and the model weights. Runs offline.

Usage (HPC):
  sbatch record_activations.sbatch
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def storage_root(cfg) -> Path:
    roots = [Path(r) for r in cfg["storage_roots"]]
    for root in roots:
        if root.is_dir():
            return root
    raise SystemExit(f"no storage root mounted; tried {roots}")


def build_prompt(tokenizer, template: str, row) -> str:
    message = template.format(question=row.question, options=row.options,
                              transcript=row.text)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], enable_thinking=False, **kwargs)
    except TypeError:  # chat template without a thinking switch
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], **kwargs)


def transcript_token_mask(offsets, rendered: str, transcript: str) -> np.ndarray:
    start = rendered.find(transcript)
    assert start != -1, "transcript not found verbatim in rendered prompt"
    end = start + len(transcript)
    return np.array([o_end > start and o_start < end for o_start, o_end in offsets])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rc = cfg["record"]
    storage = storage_root(cfg)
    acts_dir = storage / rc["acts_dir"]
    acts_dir.mkdir(parents=True, exist_ok=True)

    inputs = pd.read_parquet(storage / rc["inputs"])
    # Longest prompts first: if memory is going to fail, it fails in the
    # first minutes on the worst case, never hours in.
    length = (inputs["question"].str.len() + inputs["options"].str.len()
              + inputs["text"].str.len())
    inputs = inputs.loc[length.sort_values(ascending=False).index]
    todo = inputs[[not (acts_dir / f"{r.text_id}_{r.item}.npy").exists()
                   for r in inputs.itertuples()]]
    print(f"{len(inputs)} entities, {len(inputs) - len(todo)} already recorded, "
          f"{len(todo)} to do", flush=True)
    if todo.empty:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_dir = str(storage / rc["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, rc["dtype"]), device_map=rc["device_map"])
    model.eval()
    n_layers = model.config.num_hidden_layers
    kept = list(range(0, n_layers + 1, rc["layer_step"]))  # 0 = embeddings

    # Forward hooks pool each kept layer to (last token, transcript mean)
    # as soon as it computes, so no full hidden-state stack is ever held:
    # peak memory stays near the weights whatever the prompt length.
    current = {}   # per-row transcript token mask, set before each forward
    pooled = {}    # kept index -> (2, hidden) on cpu
    def make_hook(idx):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = current["mask"].to(h.device)
            pooled[idx] = torch.stack(
                [h[0, -1], h[0, mask].mean(0)]).float().cpu()
        return hook
    for idx in kept:  # index 0 = embeddings; index k>0 = output of layer k-1
        module = model.get_input_embeddings() if idx == 0 else model.model.layers[idx - 1]
        module.register_forward_hook(make_hook(idx))

    meta_path = acts_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(dict(
            model_dir=rc["model_dir"], layers=kept,
            poolings=["last_token", "mean_transcript"],
            save_dtype=rc["save_dtype"], prompt=rc["prompt"]), indent=2))

    n_done = n_skipped = 0
    t0 = time.time()
    for row in todo.itertuples():
        rendered = build_prompt(tokenizer, rc["prompt"], row)
        enc = tokenizer(rendered, return_offsets_mapping=True,
                        return_tensors="pt", add_special_tokens=False)
        n_tokens = enc["input_ids"].shape[1]
        if n_tokens > rc["max_input_tokens"]:
            print(f"skip {row.text_id}_{row.item}: {n_tokens} tokens", flush=True)
            n_skipped += 1
            continue
        mask = transcript_token_mask(enc["offset_mapping"][0].tolist(),
                                     rendered, row.text)
        current["mask"] = torch.from_numpy(mask)
        pooled.clear()
        with torch.no_grad():
            # Trunk only: the vocab head would build a tokens x vocab
            # logits tensor (5+ GiB on long prompts) we never use.
            # use_cache=False: a KV cache is another ~5 GiB at 20k tokens.
            model.model(input_ids=enc["input_ids"].to(model.device),
                        use_cache=False)
        acts = np.stack([pooled[i].numpy() for i in kept])
        target = acts_dir / f"{row.text_id}_{row.item}.npy"
        tmp = acts_dir / f".tmp.{row.text_id}_{row.item}.npy"
        np.save(tmp, acts.astype(rc["save_dtype"]))
        os.replace(tmp, target)
        n_done += 1
        if n_done % rc["log_every"] == 0 or n_done <= 3:
            rate = n_done / (time.time() - t0)
            peak = (max(torch.cuda.max_memory_allocated(d)
                        for d in range(torch.cuda.device_count())) / 2**30
                    if torch.cuda.is_available() else 0.0)
            print(f"{n_done}/{len(todo)} at {rate:.2f}/s, "
                  f"eta {(len(todo) - n_done) / rate / 60:.0f} min, "
                  f"peak {peak:.1f} GiB ({n_tokens} tok)", flush=True)

    print(f"done: {n_done} recorded, {n_skipped} skipped over token limit "
          f"-> {acts_dir}", flush=True)


if __name__ == "__main__":
    main()
