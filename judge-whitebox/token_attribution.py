"""Token attribution to a judge's answer-logit gap."""

import os
import pathlib

import pandas as pd
import torch

import token_manifold_record
import wbpairs
import wbprompts

STEP_BATCH_DEFAULT = 4


def answer_token_ids(tokenizer):
    """Single token ids for the two answer options."""
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in wbprompts.ANSWER_TOKENS]
    assert all(len(i) == 1 for i in ids), f"answer tokens must be single-token: {ids}"
    (id1,), (id2,) = ids
    return id1, id2


def _gap_logits(model, embeds, attention_mask, id1, id2):
    """Differentiable gap per batch row, from input embeddings."""
    out = model(inputs_embeds=embeds, attention_mask=attention_mask)
    return out.logits[:, -1, id1] - out.logits[:, -1, id2]


def attribute_tokens(model, tokenizer, input_ids, attention_mask, steps,
                     step_batch=STEP_BATCH_DEFAULT):
    """Integrated-gradient attribution of the gap to each input token.

    Interpolation points are the same sequence with different embeddings, so
    step_batch of them run as one batch. The rows are independent, so summing
    their gaps gives each row its own gradient.
    """
    assert steps > 0 and step_batch > 0
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    id1, id2 = answer_token_ids(tokenizer)

    embeds = model.get_input_embeddings()(input_ids).detach()
    baseline = torch.zeros_like(embeds)
    delta = embeds - baseline

    total = torch.zeros_like(embeds)
    for start in range(0, steps, step_batch):
        rows = min(step_batch, steps - start)
        alphas = ((torch.arange(start, start + rows, device=device, dtype=torch.float64) + 0.5)
                  / steps).view(rows, 1, 1).to(embeds.dtype)
        point = (baseline + alphas * delta).detach().requires_grad_(True)
        gaps = _gap_logits(model, point, attention_mask.expand(rows, -1), id1, id2)
        grad, = torch.autograd.grad(gaps.sum(), point)
        total += grad.sum(0, keepdim=True)

    attributions = (total / steps) * delta
    scores = attributions.sum(-1)[0].double().cpu().numpy()

    with torch.no_grad():
        gap = float(_gap_logits(model, embeds, attention_mask, id1, id2)[0])
        baseline_gap = float(_gap_logits(model, baseline, attention_mask, id1, id2)[0])
    return {"scores": scores, "gap": gap, "baseline_gap": baseline_gap,
            "completeness_error": float(scores.sum() - (gap - baseline_gap))}


def attribute_rows(model, tokenizer, df, c, steps, step_batch=STEP_BATCH_DEFAULT):
    """Yield one attribution record per rendered prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        result = attribute_tokens(model, tokenizer, enc["input_ids"],
                                  enc["attention_mask"], steps, step_batch)
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


def write_attributions(rows, out_path):
    """Write the long-format token table atomically to a new path."""
    _refuse_existing(out_path)  # fail before spending a forward pass
    groups = token_manifold_record.STRUCTURAL_GROUPS
    frames = []
    for row in rows:
        frames.append(pd.DataFrame({
            "pair_uid": row["pair_uid"], "item": row["item"], "order": row["order"],
            "position": range(len(row["scores"])), "token_id": row["input_ids"],
            "token_text": row["token_text"],
            "group": [groups[int(g)] for g in row["group_id"]],
            "score": row["scores"], "gap": row["gap"],
            "completeness_error": row["completeness_error"]}))
    return write_table(pd.concat(frames, ignore_index=True), out_path)


def select_shard(df, index, count):
    """Every count-th prompt starting at index, without reordering rows."""
    assert 0 <= index < count
    return df.iloc[index::count].copy()


def completeness_sweep(model, tokenizer, df, c, step_counts, n_prompts, seed,
                       step_batch=STEP_BATCH_DEFAULT):
    """Relative completeness error per prompt at each step count."""
    sample = df.sample(n=min(n_prompts, len(df)), random_state=seed)
    rows = []
    for steps in step_counts:
        for record in attribute_rows(model, tokenizer, sample, c, steps, step_batch):
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
    parser.add_argument("--step-batch", type=int, default=STEP_BATCH_DEFAULT,
                        help="integration steps run per forward/backward pass")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import wbconfig

    c = wbconfig.cfg()
    args = build_parser(c).parse_args()

    path = wbconfig.resolve_model(args.model, c)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                                                 device_map="auto")
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
                                       args.step_batch), args.out)
        return
    write_attributions(attribute_rows(model, tokenizer, df, c, args.steps, args.step_batch),
                       args.out)


if __name__ == "__main__":
    main()
