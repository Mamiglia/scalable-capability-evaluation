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
QUADRATURE_DEFAULT = "gauss"


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


def quadrature_points(steps, kind):
    """Sample positions along the baseline-to-prompt path, and their weights.

    Midpoint spaces samples evenly and weights them equally. Gauss-Legendre
    chooses both positions and weights so a smooth integrand needs far fewer
    samples for the same accuracy. Weights sum to one either way.
    """
    if kind == "midpoint":
        alphas = (np.arange(steps) + 0.5) / steps
        return alphas, np.full(steps, 1.0 / steps)
    if kind == "gauss":
        nodes, weights = np.polynomial.legendre.leggauss(steps)
        return (nodes + 1.0) / 2.0, weights / 2.0
    raise ValueError(f"unknown quadrature: {kind}")


def attribute_tokens(model, tokenizer, input_ids, attention_mask, steps,
                     step_batch=STEP_BATCH_DEFAULT, baseline_kind=BASELINE_DEFAULT,
                     quadrature=QUADRATURE_DEFAULT):
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
    positions, weights = quadrature_points(steps, quadrature)
    total = torch.zeros(embeds.shape, device=device, dtype=torch.float64)
    for start in range(0, steps, step_batch):
        chunk = slice(start, min(start + step_batch, steps))
        rows = len(positions[chunk])
        alphas = torch.tensor(positions[chunk], device=device,
                              dtype=torch.float64).view(rows, 1, 1).to(embeds.dtype)
        chunk_weights = torch.tensor(weights[chunk], device=device,
                                     dtype=torch.float64).view(rows, 1, 1)
        point = (baseline + alphas * delta).detach().requires_grad_(True)
        gaps = _gap_logits(model, point, attention_mask.expand(rows, -1), id1, id2)
        grad, = torch.autograd.grad(gaps.sum(), point)
        total += (grad.double() * chunk_weights).sum(0, keepdim=True)

    attributions = total * delta.double()
    scores = attributions.sum(-1)[0].cpu().numpy()
    assert np.isfinite(scores).all(), "non-finite token scores"

    with torch.no_grad():
        gap = float(_gap_logits(model, embeds, attention_mask, id1, id2)[0])
        baseline_gap = float(_gap_logits(model, baseline, attention_mask, id1, id2)[0])
    return {"scores": scores, "gap": gap, "baseline_gap": baseline_gap,
            "completeness_error": float(scores.sum() - (gap - baseline_gap))}


def _blank_setup(model, tokenizer, input_ids, attention_mask, baseline_kind):
    """Embeddings, filler row and answer ids shared by every blanking method."""
    model.requires_grad_(False)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    id1, id2 = answer_token_ids(tokenizer)
    table = model.get_input_embeddings()
    embeds = table(input_ids).detach()
    filler = (torch.zeros_like(embeds[0, 0]) if baseline_kind == "zeros"
              else table.weight.detach().mean(0))
    return embeds, attention_mask, filler, id1, id2


def _masked_deltas(model, embeds, attention_mask, draws, filler, id1, id2, batch_draws):
    """gap − gap(blanked) for each index set in draws, batched."""
    with torch.no_grad():
        gap = float(_gap_logits(model, embeds, attention_mask, id1, id2)[0])
        deltas = []
        for start in range(0, len(draws), batch_draws):
            chunk = draws[start:start + batch_draws]
            batch = embeds.expand(len(chunk), -1, -1).clone()
            for row, idx in enumerate(chunk):
                batch[row, idx] = filler
            masked = _gap_logits(model, batch, attention_mask.expand(len(chunk), -1), id1, id2)
            deltas.extend(gap - float(v) for v in masked)
    return gap, np.asarray(deltas, dtype=np.float64)


def random_mask_scores(model, tokenizer, input_ids, attention_mask, n_draws, mask_size,
                       seed, batch_draws=8, baseline_kind=BASELINE_DEFAULT):
    """Gap change when a random subset of tokens is blanked, over many draws.

    A single token rarely moves the gap far enough to measure, and its effect
    with every other token present is only its contribution at one corner of
    the input space. Blanking mask_size tokens drawn at random gives a
    measurable change and, across draws, constrains every token in many
    different contexts. Per-token effects are recovered afterwards by
    regressing the change on the draw indicators.
    """
    assert n_draws > 0 and mask_size > 0 and batch_draws > 0
    embeds, attention_mask, filler, id1, id2 = _blank_setup(
        model, tokenizer, input_ids, attention_mask, baseline_kind)
    n_tokens = embeds.shape[1]
    rng = np.random.default_rng(seed)
    draws = [rng.choice(n_tokens, size=min(mask_size, n_tokens), replace=False)
             for _ in range(n_draws)]
    gap, deltas = _masked_deltas(model, embeds, attention_mask, draws, filler,
                                 id1, id2, batch_draws)
    return {"gap": gap, "draws": draws, "deltas": deltas, "n_tokens": n_tokens}


def additivity_probe(model, tokenizer, input_ids, attention_mask, k_values, n_triples,
                     seed, batch_draws=8, baseline_kind=BASELINE_DEFAULT):
    """Blank two disjoint size-k subsets separately and jointly, per k.

    If token effects were additive, delta(S1 ∪ S2) = delta(S1) + delta(S2);
    the residual measures interaction at scale k. The singles double as the
    delta-magnitude-versus-k measurement. Sizes with 2k > T are skipped, since
    two disjoint subsets no longer fit.
    """
    assert n_triples > 0 and batch_draws > 0 and k_values
    embeds, attention_mask, filler, id1, id2 = _blank_setup(
        model, tokenizer, input_ids, attention_mask, baseline_kind)
    n_tokens = embeds.shape[1]
    rng = np.random.default_rng(seed)

    records = []
    for k in k_values:
        if 2 * k > n_tokens:
            continue
        triples = []
        for _ in range(n_triples):
            both = rng.choice(n_tokens, size=2 * k, replace=False)
            triples.append((both[:k], both[k:]))
        draws = [s for a, b in triples for s in (a, b, np.concatenate([a, b]))]
        gap, deltas = _masked_deltas(model, embeds, attention_mask, draws, filler,
                                     id1, id2, batch_draws)
        for i, (a, b) in enumerate(triples):
            d1, d2, d12 = deltas[3 * i:3 * i + 3]
            records.append({"k": k, "triple": i, "delta_a": d1, "delta_b": d2,
                            "delta_union": d12, "positions_a": a.tolist(),
                            "positions_b": b.tolist(), "gap": gap})
    return {"records": records, "n_tokens": n_tokens}


def blank_group_rows(model, tokenizer, df, c, batch_draws=8,
                     baseline_kind=BASELINE_DEFAULT):
    """Yield the gap change from blanking each whole structural group, per prompt.

    The direct joint quantity — no additive assumption — that anchors any
    aggregate of per-token scores over the same span.
    """
    groups = token_manifold_record.STRUCTURAL_GROUPS
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        embeds, attention_mask, filler, id1, id2 = _blank_setup(
            model, tokenizer, enc["input_ids"], enc["attention_mask"], baseline_kind)
        group_id = np.asarray(meta["group_id"])
        draws = [np.flatnonzero(group_id == g) for g in range(len(groups))]
        gap, deltas = _masked_deltas(model, embeds, attention_mask, draws, filler,
                                     id1, id2, batch_draws)
        yield pd.DataFrame({
            "pair_uid": row["pair_uid"], "item": row["item"], "order": int(row["order"]),
            "group": list(groups), "n_tokens": [len(d) for d in draws],
            "delta": deltas, "gap": gap})


def occlusion_rows(model, tokenizer, df, c, span_size, batch_spans=8,
                   baseline_kind=BASELINE_DEFAULT):
    """Yield one occlusion record per rendered prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        out = occlusion_scores(model, tokenizer, enc["input_ids"], enc["attention_mask"],
                               span_size, batch_spans, baseline_kind)
        groups = token_manifold_record.STRUCTURAL_GROUPS
        starts = [lo for lo, _ in out["spans"]]
        yield pd.DataFrame({
            "pair_uid": row["pair_uid"], "item": row["item"], "order": int(row["order"]),
            "span_start": starts, "span_end": [hi for _, hi in out["spans"]],
            "group": [groups[int(meta["group_id"][lo])] for lo in starts],
            "text": [tokenizer.decode(meta["input_ids"][lo:hi]) for lo, hi in out["spans"]],
            "score": out["scores"], "gap": out["gap"]})


def random_mask_rows(model, tokenizer, df, c, n_draws, mask_size, seed, batch_draws=8,
                     baseline_kind=BASELINE_DEFAULT):
    """Yield one long-format table of random-mask draws per prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        out = random_mask_scores(model, tokenizer, enc["input_ids"], enc["attention_mask"],
                                 n_draws, mask_size, seed, batch_draws, baseline_kind)
        yield pd.DataFrame({
            "pair_uid": row["pair_uid"], "item": row["item"], "order": int(row["order"]),
            "draw": range(len(out["deltas"])),
            "masked_positions": [d.tolist() for d in out["draws"]],
            "delta": out["deltas"], "gap": out["gap"], "n_tokens": out["n_tokens"]})


def additivity_rows(model, tokenizer, df, c, k_values, n_triples, seed, batch_draws=8,
                    baseline_kind=BASELINE_DEFAULT):
    """Yield one long-format table of additivity triples per prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        out = additivity_probe(model, tokenizer, enc["input_ids"], enc["attention_mask"],
                               k_values, n_triples, seed, batch_draws, baseline_kind)
        frame = pd.DataFrame(out["records"])
        frame.insert(0, "pair_uid", row["pair_uid"])
        frame.insert(1, "item", row["item"])
        frame.insert(2, "order", int(row["order"]))
        frame["n_tokens"] = out["n_tokens"]
        yield frame


def write_occlusion_parts(frames, out_dir, chunk_prompts):
    """Write occlusion tables in parts so an interrupted run can resume."""
    assert chunk_prompts > 0
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("part_*.parquet")))
    buffer, written = [], []
    for frame in frames:
        buffer.append(frame)
        if len(buffer) >= chunk_prompts:
            written.append(write_table(pd.concat(buffer, ignore_index=True),
                                       out_dir / f"part_{existing + len(written):05d}.parquet"))
            buffer = []
    if buffer:
        written.append(write_table(pd.concat(buffer, ignore_index=True),
                                   out_dir / f"part_{existing + len(written):05d}.parquet"))
    return written


def attribute_rows(model, tokenizer, df, c, steps, step_batch=STEP_BATCH_DEFAULT,
                   baseline_kind=BASELINE_DEFAULT, quadrature=QUADRATURE_DEFAULT):
    """Yield one attribution record per rendered prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        result = attribute_tokens(model, tokenizer, enc["input_ids"],
                                  enc["attention_mask"], steps, step_batch, baseline_kind,
                                  quadrature)
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
                       step_batch=STEP_BATCH_DEFAULT, baseline_kind=BASELINE_DEFAULT,
                       quadrature=QUADRATURE_DEFAULT):
    """Relative completeness error per prompt at each step count."""
    sample = df.sample(n=min(n_prompts, len(df)), random_state=seed)
    rows = []
    for steps in step_counts:
        for record in attribute_rows(model, tokenizer, sample, c, steps, step_batch,
                                     baseline_kind, quadrature):
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
    parser.add_argument("--limit-sampling", default="round_robin",
                        choices=("round_robin", "balanced"),
                        help="round_robin follows sorted pair_uid order, which concentrates "
                             "alphabetically early evaluated models; balanced caps slots per "
                             "evaluated model")
    parser.add_argument("--limit-seed", type=int, default=0,
                        help="seed for the balanced pair draw")
    parser.add_argument("--min-reasoning-chars", type=int, default=0,
                        help="drop pairs where a transcript has fewer content characters "
                             "than this beyond its answer line; 0 keeps everything")
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"),
                        help="model precision; bfloat16 matches how the judges are run")
    parser.add_argument("--step-batch", type=int, default=STEP_BATCH_DEFAULT,
                        help="integration steps run per forward/backward pass")
    parser.add_argument("--occlude-span", type=int, default=None,
                        help="measure the gap change when each span of this many tokens is "
                             "replaced by the baseline, instead of integrating gradients")
    parser.add_argument("--batch-spans", type=int, default=8)
    parser.add_argument("--mask-draws", type=int, default=None,
                        help="blank this many random token subsets per prompt, and record the "
                             "gap change; per-token effects are regressed from the draws")
    parser.add_argument("--mask-size", type=int, default=32,
                        help="tokens blanked per draw")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--blank-groups", action="store_true",
                        help="blank each whole structural group once per prompt and record "
                             "the gap change; the direct joint quantity behind span scores")
    parser.add_argument("--additivity-ks", default=None,
                        help="comma-separated subset sizes; for each, blank two disjoint "
                             "subsets separately and jointly to measure interaction at that "
                             "scale")
    parser.add_argument("--additivity-triples", type=int, default=64,
                        help="disjoint-pair draws per subset size")
    parser.add_argument("--quadrature", default=QUADRATURE_DEFAULT,
                        choices=("gauss", "midpoint"),
                        help="how the path integral is sampled")
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
    if args.min_reasoning_chars:
        pairs = wbpairs.drop_answer_only_pairs(pairs, args.min_reasoning_chars)
    folds = token_manifold_record.balanced_item_folds(pairs, args.fold_seed)
    df = token_manifold_record.select_fold(pairs, folds, args.fold)
    if args.limit_sampling == "balanced":
        df = token_manifold_record.balanced_limit_pairs(df, args.limit, args.limit_seed)
    else:
        df = token_manifold_record.limit_pairs(df, args.limit)
    df = select_shard(df, args.shard_index, args.shard_count)

    if args.sweep_steps:
        step_counts = [int(s) for s in args.sweep_steps.split(",")]
        write_table(completeness_sweep(model, tokenizer, df, c, step_counts,
                                       args.sweep_prompts, args.sweep_seed,
                                       args.step_batch, args.baseline,
                                       args.quadrature), args.out)
        return
    if args.blank_groups:
        done = completed_prompts(args.out) if pathlib.Path(args.out).exists() else set()
        if done:
            keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
            print(f"resuming: {len(done)} done, {sum(keep)} to go", flush=True)
            df = df[keep]
        if df.empty:
            print("nothing left to blank", flush=True)
            return
        write_occlusion_parts(blank_group_rows(model, tokenizer, df, c, args.batch_spans,
                                               args.baseline),
                              args.out, args.chunk_prompts)
        return

    if args.additivity_ks:
        k_values = [int(k) for k in args.additivity_ks.split(",")]
        done = completed_prompts(args.out) if pathlib.Path(args.out).exists() else set()
        if done:
            keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
            print(f"resuming: {len(done)} done, {sum(keep)} to go", flush=True)
            df = df[keep]
        if df.empty:
            print("nothing left to probe", flush=True)
            return
        write_occlusion_parts(additivity_rows(model, tokenizer, df, c, k_values,
                                              args.additivity_triples, args.mask_seed,
                                              args.batch_spans, args.baseline),
                              args.out, args.chunk_prompts)
        return

    if args.mask_draws:
        done = completed_prompts(args.out) if pathlib.Path(args.out).exists() else set()
        if done:
            keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
            print(f"resuming: {len(done)} done, {sum(keep)} to go", flush=True)
            df = df[keep]
        if df.empty:
            print("nothing left to mask", flush=True)
            return
        write_occlusion_parts(random_mask_rows(model, tokenizer, df, c, args.mask_draws,
                                               args.mask_size, args.mask_seed,
                                               args.batch_spans, args.baseline),
                              args.out, args.chunk_prompts)
        return

    if args.occlude_span:
        done = completed_prompts(args.out) if pathlib.Path(args.out).exists() else set()
        if done:
            keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
            print(f"resuming: {len(done)} done, {sum(keep)} to go", flush=True)
            df = df[keep]
        if df.empty:
            print("nothing left to occlude", flush=True)
            return
        write_occlusion_parts(occlusion_rows(model, tokenizer, df, c, args.occlude_span,
                                             args.batch_spans, args.baseline),
                              args.out, args.chunk_prompts)
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
                                           args.step_batch, args.baseline, args.quadrature),
                            args.out, args.chunk_prompts)


def occlusion_scores(model, tokenizer, input_ids, attention_mask, span_size,
                     batch_spans=8, baseline_kind=BASELINE_DEFAULT):
    """Change in the gap when each span of tokens is replaced by the baseline embedding.

    A score is the model's own answer to "what happens without this", measured
    rather than approximated, so there is no integral to converge and no
    completeness identity to satisfy. Spans of one token give the finest
    resolution; larger spans cost proportionally less and register effects too
    small to see one token at a time.
    """
    assert span_size > 0 and batch_spans > 0
    model.requires_grad_(False)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    id1, id2 = answer_token_ids(tokenizer)

    table = model.get_input_embeddings()
    embeds = table(input_ids).detach()
    filler = (torch.zeros_like(embeds[0, 0]) if baseline_kind == "zeros"
              else table.weight.detach().mean(0))
    n_tokens = embeds.shape[1]
    spans = [(s, min(s + span_size, n_tokens)) for s in range(0, n_tokens, span_size)]

    with torch.no_grad():
        gap = float(_gap_logits(model, embeds, attention_mask, id1, id2)[0])
        scores = []
        for start in range(0, len(spans), batch_spans):
            chunk = spans[start:start + batch_spans]
            batch = embeds.expand(len(chunk), -1, -1).clone()
            for row, (lo, hi) in enumerate(chunk):
                batch[row, lo:hi] = filler
            occluded = _gap_logits(model, batch, attention_mask.expand(len(chunk), -1), id1, id2)
            scores.extend(gap - float(v) for v in occluded)
    return {"gap": gap, "spans": spans, "scores": np.asarray(scores, dtype=np.float64)}


if __name__ == "__main__":
    main()
