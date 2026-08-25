"""Prefix traces of the shared SAE features for the viewer's example transcripts.

The stored embedding is the final-layer state at the end token appended after the
transcript. For a set of boundary positions t, the embedding of the prefix [0, t) is
obtained exactly in one forward pass: append one end token per boundary, give it
position t, and mask it to attend only to tokens [0, t) and itself. Each prefix
embedding is item-centred and projected onto the shared encoder rows (signed
pre-activation). The full-length boundary reproduces the stored activation, which is
checked.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcript_embeddings import special_token_frame  # noqa: E402

KEY = ["bench", "item", "model", "epoch", "log"]


def item_means(dense_dir):
    """Mean raw transcript-only embedding per item, keyed by A's transcript keys."""
    rows = pd.read_parquet(dense_dir / "rows.parquet", columns=["transcript_key", "item_group_key"])
    audit = pd.read_parquet(dense_dir / "audit_rows.parquet")
    if not rows.transcript_key.equals(audit.transcript_key):
        raise ValueError("dense rows and audit rows are not aligned")
    dense = np.load(dense_dir / "transcript_only__final.npy")
    means = {item: dense[index].mean(axis=0) for item, index in rows.groupby("item_group_key").indices.items()}
    key_to_item = dict(zip(zip(*(audit[k] for k in KEY)), rows.item_group_key))
    return means, key_to_item


class SharedEncoder:
    """Signed pre-activation of the shared SAE's features, with the stored sign convention."""

    def __init__(self, sae_dir):
        state = torch.load(sae_dir / "sae__shared.pt", map_location="cpu")["state_dict"]
        self.input_bias = state["input_bias"].numpy()
        self.weight = state["encoder.weight"].numpy()
        self.neuron_bias = state["neuron_bias"].numpy()
        self.threshold = float(state["threshold"])
        report = json.loads((sae_dir / "sae_report.json").read_text())
        flipped = np.array(report["groups"]["shared"]["sign_flipped"])
        self.sign = np.where(flipped, -1.0, 1.0).astype(np.float32)

    def __call__(self, centred):
        return ((centred - self.input_bias) @ self.weight.T + self.neuron_bias) * self.sign


def centre(hidden, mean):
    centred = hidden - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    return np.where(norms > 0, centred / np.where(norms > 0, norms, 1), 0.0).astype(np.float32)


def boundaries(n_tokens, max_boundaries):
    """Prefix ends 1..n, thinned to at most max_boundaries evenly spaced, always ending at n."""
    if n_tokens <= max_boundaries:
        return list(range(1, n_tokens + 1))
    return sorted(set(np.linspace(1, n_tokens, max_boundaries).round().astype(int).tolist()) | {n_tokens})


def prefix_mask(n_tokens, bounds):
    """Causal mask over the transcript; end token j sees tokens [0, bounds[j]) and itself."""
    total = n_tokens + len(bounds)
    mask = torch.zeros(total, total, dtype=torch.bool)
    mask[:n_tokens, :n_tokens] = torch.tril(torch.ones(n_tokens, n_tokens, dtype=torch.bool))
    for j, t in enumerate(bounds):
        mask[n_tokens + j, :t] = True
        mask[n_tokens + j, n_tokens + j] = True
    return mask


def prefix_embeddings(model, tokenizer, ids, bounds, device):
    """Unit-norm embedding of every prefix in one pass, in the order of bounds."""
    prefix, suffix = special_token_frame(tokenizer)
    if prefix:
        raise ValueError("a non-empty special-token prefix is not handled")
    end = suffix[0]
    n = len(ids)
    tokens = torch.tensor([ids + [end] * len(bounds)], device=device)
    positions = torch.tensor([list(range(n)) + list(bounds)], device=device)
    mask = prefix_mask(n, bounds)[None, None].to(device)
    with torch.inference_mode():
        hidden = model(input_ids=tokens, attention_mask=mask, position_ids=positions).last_hidden_state[0, n:]
    return torch.nn.functional.normalize(hidden.float(), dim=1).cpu().numpy()


def collect_examples(viewer_data):
    """Every example transcript across the feature files: key -> (text, {feature: stored activation})."""
    wanted = {}
    for path in sorted(viewer_data.glob("feature_*.json")):
        payload = json.loads(path.read_text())
        for group in payload["groups"].values():
            for example in group["examples"]:
                key = "|".join(str(example[k]) for k in KEY)
                wanted.setdefault(key, (example["transcript"], {}))[1][payload["feature"]] = example["activation"]
    return wanted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-data", type=Path, default=ROOT / "interactive_dashboards/sae_features/data")
    parser.add_argument("--dense-dir", type=Path, default=ROOT / "judge_results/verdict_dissection/transcript_sae/dense-qwen")
    parser.add_argument("--sae-dir", type=Path, default=ROOT / "judge_results/verdict_dissection/transcript_sae/sae-qwen")
    parser.add_argument("--model-path", type=Path, default=Path("/Volumes/T7/models/Qwen3-Embedding-4B-5cf2132a"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-tokens", type=int, default=6144, help="skip longer transcripts")
    parser.add_argument("--max-boundaries", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="trace only the first N transcripts (smoke)")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(args.device).eval()
    means, key_to_item = item_means(args.dense_dir)
    encoder = SharedEncoder(args.sae_dir)
    n_features = encoder.weight.shape[0]

    out_dir = args.viewer_data / "traces"
    out_dir.mkdir(exist_ok=True)
    traces = {}
    for feature in range(n_features):
        path = out_dir / f"feature_{feature:02d}.json"
        traces[feature] = json.loads(path.read_text()) if path.exists() else {}
    wanted = collect_examples(args.viewer_data)
    keys = [k for k in wanted if any(k not in traces[f] for f in wanted[k][1])]
    keys = keys[: args.limit] if args.limit else keys
    print(f"{len(wanted)} example transcripts, {len(keys)} still to trace", flush=True)

    checks, skipped = [], 0
    for n, key in enumerate(keys, 1):
        text, stored = wanted[key]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) > args.max_tokens or not ids:
            skipped += 1
            continue
        bounds = boundaries(len(ids), args.max_boundaries)
        embeddings = prefix_embeddings(model, tokenizer, ids, bounds, args.device)
        item = key_to_item[tuple(key.split("|")[:3] + [int(key.split("|")[3])] + key.split("|")[4:])]
        pre = encoder(centre(embeddings, means[item]))  # (boundaries, features)
        spans = [[offsets[previous][0], offsets[t - 1][1]] for previous, t in zip([0] + bounds[:-1], bounds)]
        for feature, activation in stored.items():
            traces[feature][key] = {"spans": spans, "values": pre[:, feature].round(4).tolist(),
                                    "final": float(pre[-1, feature])}
            if activation != 0:
                checks.append(abs(float(pre[-1, feature]) - activation))
        if n % 250 == 0 or n == len(keys):
            print(f"traced {n}/{len(keys)}, skipped {skipped}, "
                  f"max |final - stored| {max(checks) if checks else float('nan'):.4f}", flush=True)
            for feature, values in traces.items():
                (out_dir / f"feature_{feature:02d}.json").write_text(json.dumps(values))
    print(f"done: {len(keys) - skipped} traced, {skipped} skipped (> {args.max_tokens} tokens); "
          f"threshold {encoder.threshold:.4f}; max |final - stored| over nonzero stored "
          f"{max(checks) if checks else float('nan'):.4f}")


if __name__ == "__main__":
    main()
