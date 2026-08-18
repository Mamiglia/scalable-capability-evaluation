"""Token attribution to a judge's answer-logit gap."""

import os
import pathlib

import numpy as np
import pandas as pd
import torch

import token_manifold_record
import wbpairs
import wbprompts

STEP_BATCH_DEFAULT = 4
BASELINE_DEFAULT = "mean"


def answer_token_ids(tokenizer):
    """Single token ids for the two answer options."""
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in wbprompts.ANSWER_TOKENS]
    assert all(len(i) == 1 for i in ids), f"answer tokens must be single-token: {ids}"
    (id1,), (id2,) = ids
    return id1, id2


def _gap_logits(model, embeds, attention_mask, id1, id2):
    """Differentiable gap per batch row, from input embeddings.

    Only the last position's two answer logits are needed, so the final hidden
    state is projected onto the difference of their unembedding rows. Running
    the full head instead would materialise vocabulary-wide logits at every
    position, several gigabytes per forward, all of it kept for the backward.
    """
    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None, "unembedding bias would break this shortcut"
    hidden = model.get_decoder()(inputs_embeds=embeds, attention_mask=attention_mask)
    return hidden.last_hidden_state[:, -1] @ (head.weight[id1] - head.weight[id2])


def attribute_tokens(model, tokenizer, input_ids, attention_mask, steps,
                     step_batch=STEP_BATCH_DEFAULT, baseline_kind=BASELINE_DEFAULT):
    """Integrated-gradient attribution of the gap to each input token.

    Interpolation points are the same sequence with different embeddings, so
    step_batch of them run as one batch. The rows are independent, so summing
    their gaps gives each row its own gradient.
    """
    assert steps > 0 and step_batch > 0
    # Only the input embeddings need gradients; tracking the weights as well
    # buffers parameter gradients that are never read and exhausts the GPU.
    model.requires_grad_(False)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    assert int(attention_mask[0, -1]) == 1, "the gap is read at the last position, which is padding"
    id1, id2 = answer_token_ids(tokenizer)

    table = model.get_input_embeddings()
    embeds = table(input_ids).detach()
    if baseline_kind == "zeros":
        baseline = torch.zeros_like(embeds)
    elif baseline_kind == "mean":
        # Zeros are degenerate under RMSNorm: the normalised stream stays put
        # until alpha is almost zero, then the integrand spikes and the integral
        # needs impractically many steps. The mean token embedding has a typical
        # norm, so the path is smooth.
        baseline = table.weight.detach().mean(0).expand_as(embeds).contiguous()
    else:
        raise ValueError(f"unknown baseline: {baseline_kind}")
    delta = embeds - baseline

    # Accumulate and reduce in float64. Summing thousands of gradients and then
    # 4096 embedding dimensions in float32 loses small and cancelling terms, and
    # that error would land in the completeness figure as if it were quadrature
    # error.
    total = torch.zeros(embeds.shape, device=device, dtype=torch.float64)
    for start in range(0, steps, step_batch):
        rows = min(step_batch, steps - start)
        alphas = ((torch.arange(start, start + rows, device=device, dtype=torch.float64) + 0.5)
                  / steps).view(rows, 1, 1).to(embeds.dtype)
        point = (baseline + alphas * delta).detach().requires_grad_(True)
        gaps = _gap_logits(model, point, attention_mask.expand(rows, -1), id1, id2)
        grad, = torch.autograd.grad(gaps.sum(), point)
        total += grad.double().sum(0, keepdim=True)

    attributions = (total / steps) * delta.double()
    scores = attributions.sum(-1)[0].cpu().numpy()
    assert np.isfinite(scores).all(), "non-finite token scores"

    with torch.no_grad():
        gap = float(_gap_logits(model, embeds, attention_mask, id1, id2)[0])
        baseline_gap = float(_gap_logits(model, baseline, attention_mask, id1, id2)[0])
    return {"scores": scores, "gap": gap, "baseline_gap": baseline_gap,
            "completeness_error": float(scores.sum() - (gap - baseline_gap))}


def attribute_rows(model, tokenizer, df, c, steps, step_batch=STEP_BATCH_DEFAULT,
                   baseline_kind=BASELINE_DEFAULT):
    """Yield one attribution record per rendered prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        result = attribute_tokens(model, tokenizer, enc["input_ids"],
                                  enc["attention_mask"], steps, step_batch, baseline_kind)
        assert len(result["scores"]) == len(meta["input_ids"])
        yield {"pair_uid": row["pair_uid"], "item": row["item"], "order": int(row["order"]),
               "gap": result["gap"], "baseline_gap": result["baseline_gap"],
               "completeness_error": result["completeness_error"],
               "scores": result["scores"], "input_ids": list(meta["input_ids"]),
               "offsets": list(meta["offsets"]), "group_id": meta["group_id"],
               "token_text": [tokenizer.decode([i]) for i in meta["input_ids"]]}


def _refuse_existing(out_path):
    """Destination and its temporary file, refusing to overwrite either."""
    out_path = pathlib.Path(out_path)
    tmp = out_path.with_name(out_path.name + ".partial")
    if out_path.exists() or tmp.exists():
        raise FileExistsError(out_path if out_path.exists() else tmp)
    return out_path, tmp


def write_table(table, out_path):
    """Write a parquet to a new path through a temporary file and a rename."""
    out_path, tmp = _refuse_existing(out_path)
    table.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return out_path


def prompt_frame(row):
    """One prompt's attributions as long-format rows, one per token."""
    groups = token_manifold_record.STRUCTURAL_GROUPS
    return pd.DataFrame({
        "pair_uid": row["pair_uid"], "item": row["item"], "order": row["order"],
        "position": range(len(row["scores"])), "token_id": row["input_ids"],
        "token_text": row["token_text"],
        "group": [groups[int(g)] for g in row["group_id"]],
        "char_start": [int(start) for start, _ in row["offsets"]],
        "char_end": [int(end) for _, end in row["offsets"]],
        "score": row["scores"], "gap": row["gap"],
        "baseline_gap": row["baseline_gap"],
        "completeness_error": row["completeness_error"]})


def write_attributions(rows, out_path):
    """Write the long-format token table atomically to a new path."""
    _refuse_existing(out_path)  # fail before spending a forward pass
    frames = [prompt_frame(row) for row in rows]
    assert frames, "no prompts to attribute; check the split, limit and shard count"
    return write_table(pd.concat(frames, ignore_index=True), out_path)


def completed_prompts(out_dir):
    """Prompts already written to a part file, as (pair_uid, order)."""
    out_dir = pathlib.Path(out_dir)
    done = set()
    for part in sorted(out_dir.glob("part_*.parquet")):
        table = pd.read_parquet(part, columns=["pair_uid", "order"]).drop_duplicates()
        done.update((str(uid), int(order)) for uid, order in table.to_numpy())
    return done


def write_attribution_parts(rows, out_dir, chunk_prompts):
    """Write attributions in numbered parts so an interrupted run can resume.

    A long run on metered hardware must not lose everything to one failure, so
    each chunk lands on disk before the next is computed.
    """
    assert chunk_prompts > 0
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("part_*.parquet")))
    buffer, written = [], []
    for row in rows:
        buffer.append(prompt_frame(row))
        if len(buffer) >= chunk_prompts:
            written.append(write_table(pd.concat(buffer, ignore_index=True),
                                       out_dir / f"part_{existing + len(written):05d}.parquet"))
            buffer = []
    if buffer:
        written.append(write_table(pd.concat(buffer, ignore_index=True),
                                   out_dir / f"part_{existing + len(written):05d}.parquet"))
    return written


def read_attribution_parts(out_dir):
    """Every part in one table, for analysis once a run has finished."""
    parts = sorted(pathlib.Path(out_dir).glob("part_*.parquet"))
    assert parts, f"no parts in {out_dir}"
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def select_shard(df, index, count):
    """Every count-th prompt starting at index, without reordering rows."""
    assert 0 <= index < count
    return df.iloc[index::count].copy()


def completeness_sweep(model, tokenizer, df, c, step_counts, n_prompts, seed,
                       step_batch=STEP_BATCH_DEFAULT, baseline_kind=BASELINE_DEFAULT):
    """Relative completeness error per prompt at each step count."""
    sample = df.sample(n=min(n_prompts, len(df)), random_state=seed)
    rows = []
    for steps in step_counts:
        for record in attribute_rows(model, tokenizer, sample, c, steps, step_batch,
                                     baseline_kind):
            span = abs(record["gap"] - record["baseline_gap"])
            rows.append({"steps": steps, "pair_uid": record["pair_uid"],
                         "order": record["order"],
                         "relative_error": abs(record["completeness_error"]) / max(span, 1e-12)})
    return pd.DataFrame(rows)


def build_parser(c):
    """Command-line arguments, with defaults taken from the config."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", required=True, choices=("A", "B", "all"))
    parser.add_argument("--fold-seed", type=int,
                        default=c.get("token_manifold", {}).get("fold_seed", 0))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"),
                        help="model precision; bfloat16 matches how the judges are run")
    parser.add_argument("--step-batch", type=int, default=STEP_BATCH_DEFAULT,
                        help="integration steps run per forward/backward pass")
    parser.add_argument("--baseline", default=BASELINE_DEFAULT, choices=("mean", "zeros"),
                        help="integrated-gradient baseline; zeros is degenerate under RMSNorm")
    parser.add_argument("--chunk-prompts", type=int, default=20,
                        help="prompts per part file; a part on disk survives an interruption")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split the prompts across this many jobs")
    parser.add_argument("--sweep-steps", default=None,
                        help="comma-separated step counts; writes a completeness sweep "
                             "instead of an attribution table")
    parser.add_argument("--sweep-prompts", type=int, default=20)
    parser.add_argument("--sweep-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser


def main():
    import verdicts
    import wbconfig

    c = wbconfig.cfg()
    args = build_parser(c).parse_args()

    # Same loader the judging and the recordings use, so attribution explains
    # the model that actually produces the verdicts rather than a higher
    # precision copy of it.
    dtype = getattr(torch, args.dtype)
    model, tokenizer, _ = verdicts.load_model(args.model, None, c, dtype=dtype)
    model.eval()

    pairs = wbpairs.load_pairs(c, args.split)
    folds = token_manifold_record.balanced_item_folds(pairs, args.fold_seed)
    df = token_manifold_record.select_fold(pairs, folds, args.fold)
    df = token_manifold_record.limit_pairs(df, args.limit)
    df = select_shard(df, args.shard_index, args.shard_count)

    if args.sweep_steps:
        step_counts = [int(s) for s in args.sweep_steps.split(",")]
        write_table(completeness_sweep(model, tokenizer, df, c, step_counts,
                                       args.sweep_prompts, args.sweep_seed,
                                       args.step_batch, args.baseline), args.out)
        return
    done = completed_prompts(args.out) if pathlib.Path(args.out).exists() else set()
    if done:
        keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
        print(f"resuming: {len(done)} prompts already written, {sum(keep)} to go", flush=True)
        df = df[keep]
    if df.empty:
        print("nothing left to attribute", flush=True)
        return
    write_attribution_parts(attribute_rows(model, tokenizer, df, c, args.steps,
                                           args.step_batch, args.baseline),
                            args.out, args.chunk_prompts)


if __name__ == "__main__":
    main()
