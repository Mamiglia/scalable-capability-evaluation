"""Per-token residual recording for output-conditioned manifold discovery."""

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

import token_ablate
import verdicts
import wbconfig
import wbpairs
import wbprompts


STRUCTURAL_GROUPS = ("instr_format", "slot_labels", "question", "transcript_1",
                     "transcript_2", "answer_readout")
LIMIT_SAMPLING = "item_round_robin_v1"


def balanced_item_folds(df, seed):
    """Deterministically split whole items while greedily balancing row counts."""
    counts = df.groupby("item", sort=True).size()
    rng = np.random.default_rng(seed)
    tie = dict(zip(counts.index, rng.random(len(counts))))
    ordered = sorted(counts.index, key=lambda item: (-int(counts[item]), tie[item], str(item)))
    totals = {"A": 0, "B": 0}
    folds = {}
    for item in ordered:
        fold = min(totals, key=lambda name: (totals[name], name))
        folds[item] = fold
        totals[fold] += int(counts[item])
    return folds


def select_fold(df, folds, name):
    """Return one fold, or every item when name is "all", without reordering rows."""
    assert name in {"A", "B", "all"}
    assert set(df["item"]) <= set(folds)
    if name == "all":
        return df.copy()
    return df[df["item"].map(folds) == name].copy()


def limit_pairs(df, limit):
    """Limit a selected fold by whole pair IDs without reordering rows."""
    if limit is None:
        return df.copy()
    pairs = df[["pair_uid", "item"]].drop_duplicates()
    if pairs.groupby("pair_uid")["item"].nunique().max() != 1:
        raise ValueError("pair_uid must map to exactly one item")
    by_item = {
        item: sorted(group["pair_uid"].tolist(), key=str)
        for item, group in pairs.groupby("item", sort=True)
    }
    target = min(limit, len(pairs))
    selected, round_idx = [], 0
    while len(selected) < target:
        added = False
        for item in sorted(by_item, key=str):
            if round_idx < len(by_item[item]):
                selected.append(by_item[item][round_idx])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        round_idx += 1
    return df[df["pair_uid"].isin(selected)].copy()


def layer_indices(n_states, n_select=12):
    """Evenly spaced hidden-state indices including both endpoints."""
    assert n_states > 0 and n_select > 0
    points = np.rint(np.linspace(0, n_states - 1, min(n_states, n_select))).astype(int)
    return tuple(sorted(set(points.tolist())))


def prompt_token_meta(row, tokenizer, c):
    """Aligned token IDs, offsets, and one structural group ID per token."""
    prompt = wbprompts.render_verdict_prompt(row, tokenizer)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    groups = token_ablate.token_groups(row, tokenizer, c)
    group_id = np.full(len(enc["input_ids"]), -1, dtype=np.int8)
    for idx, name in enumerate(STRUCTURAL_GROUPS):
        group_id[groups[name]] = idx
    assert (group_id >= 0).all(), "structural token groups must cover the prompt"
    return {"prompt": prompt, "input_ids": enc["input_ids"],
            "offsets": enc["offset_mapping"], "group_id": group_id,
            "group_names": STRUCTURAL_GROUPS}


def estimate_storage(df, tokenizer, c, layers, d_model, dtype_bytes=2):
    """Exact residual-byte estimate from rendered token counts; writes nothing."""
    n_tokens = sum(len(tokenizer(wbprompts.render_verdict_prompt(row, tokenizer))["input_ids"])
                   for _, row in df.iterrows())
    return {"n_prompts": len(df), "n_pairs": int(df["pair_uid"].nunique()),
            "n_items": int(df["item"].nunique()), "n_tokens": n_tokens,
            "n_layers": len(layers),
            "d_model": d_model,
            "residual_bytes": n_tokens * len(layers) * d_model * dtype_bytes,
            "metadata_bytes_upper": n_tokens * 32}


def record_rows(model, tokenizer, df, c, layers):
    """Yield one float16 `[tokens, selected layers, hidden]` record per prompt."""
    device = next(model.parameters()).device
    for row_idx, row in df.iterrows():
        meta = prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        enc = {name: value.to(device) for name, value in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        assert max(layers) < len(out.hidden_states)
        resid = torch.stack([out.hidden_states[layer][0] for layer in layers], dim=1)
        assert resid.shape[0] == len(meta["input_ids"])
        l1, l2 = verdicts.readout(out.logits[0, -1], tokenizer)
        yield {"row_idx": row_idx, "pair_uid": row["pair_uid"], "item": row["item"],
               "order": int(row["order"]), "resid": resid.float().cpu().half(),
               "input_ids": list(meta["input_ids"]), "offsets": list(meta["offsets"]),
               "group_id": np.asarray(meta["group_id"], dtype=np.int8),
               "group_names": tuple(meta["group_names"]),
               "l1": l1, "l2": l2, "gap": l1 - l2}


def sha256(path):
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_identity(df, tokenizer, pairs_path, model_revision):
    """Hashes that pin a recording to its pairs file, prompts, and model revision."""
    prompt_digest = hashlib.sha256()
    for _, row in df.iterrows():
        prompt_digest.update(wbprompts.render_verdict_prompt(row, tokenizer).encode())
        prompt_digest.update(b"\0")
    return {"pairs_sha256": sha256(pairs_path), "prompt_sha256": prompt_digest.hexdigest(),
            "model_revision": model_revision}


def atomic_torch_save(obj, path):
    """Write a new torch file atomically; never replace a final or partial file."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".partial")
    if path.exists() or tmp.exists():
        raise FileExistsError(path if path.exists() else tmp)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_chunk(chunk, path):
    prompt_idx = torch.cat([torch.full((len(row["input_ids"]),), i, dtype=torch.int32)
                            for i, row in enumerate(chunk)])
    blob = {
        "resid": torch.cat([row["resid"] for row in chunk]),
        "prompt_idx": prompt_idx,
        "position": torch.cat([torch.arange(len(row["input_ids"]), dtype=torch.int32)
                               for row in chunk]),
        "input_ids": torch.tensor([token for row in chunk for token in row["input_ids"]],
                                  dtype=torch.int32),
        "offsets": torch.tensor([offset for row in chunk for offset in row["offsets"]],
                                dtype=torch.int32),
        "group_id": torch.tensor(np.concatenate([row["group_id"] for row in chunk]),
                                 dtype=torch.int8),
        "prompts": [{key: row[key] for key in ("row_idx", "pair_uid", "item", "order",
                                                "l1", "l2", "gap")}
                    for row in chunk],
    }
    atomic_torch_save(blob, path)
    return len(prompt_idx)


def write_shards(rows, out_dir, shard_prompts, manifest_meta):
    """Write immutable shards, resuming a verified completed prefix after interruption."""
    out_dir = pathlib.Path(out_dir)
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(out_dir)
    assert shard_prompts > 0
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("shard_*.pt"))
    expected = [out_dir / f"shard_{idx:05d}.pt" for idx in range(len(existing))]
    if existing != expected:
        raise ValueError(f"existing shards are not a contiguous prefix: {out_dir}")
    entries, chunk, n_prompts, n_tokens = [], [], 0, 0

    def flush():
        nonlocal chunk, n_tokens
        if not chunk:
            return
        name = f"shard_{len(entries):05d}.pt"
        path = out_dir / name
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            identity_keys = ("row_idx", "pair_uid", "item", "order")
            expected_ids = [tuple(row[key] for key in identity_keys) for row in chunk]
            actual_ids = [tuple(prompt[key] for key in identity_keys)
                          for prompt in blob["prompts"]]
            if actual_ids != expected_ids:
                raise ValueError(f"existing shard does not match input rows: {path}")
            count = len(blob["prompt_idx"])
        else:
            count = _write_chunk(chunk, path)
        n_tokens += count
        entries.append({"path": name, "sha256": sha256(path), "n_prompts": len(chunk),
                        "n_tokens": count, "nbytes": path.stat().st_size})
        chunk = []

    for row in rows:
        chunk.append(row)
        n_prompts += 1
        if len(chunk) == shard_prompts:
            flush()
    flush()
    if len(entries) < len(existing):
        raise ValueError(f"existing shard prefix is longer than input rows: {out_dir}")

    manifest = {**manifest_meta, "version": 1, "n_prompts": n_prompts,
                "n_tokens": n_tokens, "shards": entries}
    path = out_dir / "manifest.json"
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load_tokenizer_and_shape(model_name, c):
    """Load only tokenizer/config for a storage estimate."""
    model_path = wbconfig.resolve_model(model_name, c)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    return tokenizer, int(config.hidden_size), int(config.num_hidden_layers) + 1


def _parse_layers(value, n_states):
    if value == "auto":
        return layer_indices(n_states)
    layers = tuple(int(part) for part in value.split(","))
    assert layers and min(layers) >= 0 and max(layers) < n_states
    return layers


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", required=True, choices=("A", "B", "all"))
    parser.add_argument("--fold-seed", type=int,
                        default=c.get("token_manifold", {}).get("fold_seed", 0))
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--shard-prompts", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    df = wbpairs.load_pairs(c, args.split, None)
    folds = balanced_item_folds(df, args.fold_seed)
    df = select_fold(df, folds, args.fold)
    df = limit_pairs(df, args.limit)

    if args.estimate_only:
        tokenizer, d_model, n_states = load_tokenizer_and_shape(args.model, c)
        layers = _parse_layers(args.layers, n_states)
        print(json.dumps(estimate_storage(df, tokenizer, c, layers, d_model), sort_keys=True))
        return

    model, tokenizer, _ = verdicts.load_model(args.model, None, c)
    layers = _parse_layers(args.layers, int(model.config.num_hidden_layers) + 1)
    identity = manifest_identity(df, tokenizer, c["paths"]["pairs"],
                                 getattr(model.config, "_name_or_path", args.model))
    manifest = write_shards(
        record_rows(model, tokenizer, df, c, layers), args.out, args.shard_prompts,
        {"model": args.model, "layers": list(layers), "fold": args.fold,
         "fold_seed": args.fold_seed, "split": args.split, "limit": args.limit,
         "limit_sampling": LIMIT_SAMPLING,
         "pairs_path": c["paths"]["pairs"], **identity})
    print(f"token recording -> {manifest}")


if __name__ == "__main__":
    main()
