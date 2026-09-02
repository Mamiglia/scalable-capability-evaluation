"""Projection ablation at a decoder layer: gap change, answer KL, prompt NLL.

Each condition removes a fixed direction from the residual stream at one
layer's output, at a dose: h <- h - dose * (h.u) u. Conditions sharing a layer
run as one batched forward with per-row directions. Per pass, the verdict
logit gap, the KL of the answer-position next-token distribution against the
baseline pass, and the mean next-token NLL over the prompt measure effect and
selectivity.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import torch

import token_attribution
import token_manifold_record
import wbpairs


def load_directions(npz_path):
    """Ordered {name: unit vector} from the frozen-direction archive."""
    with np.load(npz_path) as z:
        out = {name: z[name].astype(np.float32) for name in z.files}
    for name, u in out.items():
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-4, name
    return out


def condition_table(directions, fit_layer, doses, control_layer, n_random, seed,
                    restrictions=("full",)):
    """One row per pass: baseline, fitted directions at doses, layer control,
    random unit directions. The random draws are fixed by the seed.

    A restriction limits which positions the projection touches: "full" (every
    position), "not_last" (all but the final token), "role" (only the tokens of
    the direction's own role), "not_role" (every position but those)."""
    d = len(next(iter(directions.values())))
    rows = [{"name": "baseline", "layer": 0, "dose": 0.0, "vector": None,
             "restriction": "full"}]
    for name, u in directions.items():
        for restriction in restrictions:
            for dose in doses:
                rows.append({"name": name, "layer": fit_layer, "dose": float(dose),
                             "vector": u, "restriction": restriction})
        rows.append({"name": name, "layer": control_layer, "dose": 1.0, "vector": u,
                     "restriction": restrictions[0]})
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        v = rng.normal(size=d).astype(np.float32)
        rows.append({"name": f"random_{i}", "layer": fit_layer, "dose": 1.0,
                     "vector": v / np.linalg.norm(v), "restriction": "full"})
    return rows


def role_token_mask(meta, role_name):
    """Boolean per token: does it belong to the named role in this prompt?

    Roles as in the direction fitting: scaffolding groups whole, transcript
    sentences binned into position deciles within their transcript.
    """
    import mask_regression
    import token_mediation

    unit_ids = mask_regression.sentence_units(meta)
    gid = np.asarray(meta["group_id"])
    groups = meta["group_names"]
    first_token = [int(np.flatnonzero(unit_ids == u)[0])
                   for u in range(int(unit_ids.max()) + 1)]
    frame = pd.DataFrame({"pair_uid": "p", "order": 0,
                          "unit": range(len(first_token)),
                          "group": [groups[int(gid[t])] for t in first_token]})
    roles, names = token_mediation.role_ids(frame)
    if role_name not in names:
        return np.zeros(len(unit_ids), dtype=bool)
    role_units = frame.unit[np.asarray(roles) == names.index(role_name)].to_numpy()
    return np.isin(unit_ids, role_units)


def position_mask(meta, restriction, role_name, n_tokens):
    """Float per token: 1 where the projection applies under the restriction."""
    if restriction == "full":
        mask = np.ones(n_tokens)
    elif restriction == "not_last":
        mask = np.ones(n_tokens)
        mask[-1] = 0.0
    elif restriction in ("role", "not_role"):
        # a short transcript may lack the decile entirely; the mask is then
        # empty (role) or full (not_role) and n_touched records it
        role = role_token_mask(meta, role_name).astype(float)
        mask = role if restriction == "role" else 1.0 - role
    else:
        raise ValueError(f"unknown restriction: {restriction}")
    return mask


def _projection_hook(u, dose, mask=None):
    """Forward hook removing dose*(h.u)u per batch row; u [rows, d], dose [rows],
    mask [rows, positions] limiting which positions are touched (None = all)."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        coef = torch.einsum("bsd,bd->bs", h, u) * dose[:, None]
        if mask is not None:
            coef = coef * mask
        h = h - coef[:, :, None] * u[:, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def _nll_per_row(hidden, head_weight, input_ids, chunk=512):
    """Mean next-token NLL per batch row."""
    total = torch.zeros(hidden.shape[0], device=hidden.device)
    n = hidden.shape[1] - 1
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        logits = hidden[:, lo:hi].float() @ head_weight.T.float()
        target = input_ids[:, lo + 1:hi + 1]
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), target, reduction="none")
        total += loss.sum(dim=1)
    return (total / n).tolist()


def run_prompt(model, tokenizer, input_ids, attention_mask, conditions, batch=4,
               meta=None):
    """All conditions for one prompt; returns per-condition gap, answer KL, NLL."""
    device = next(model.parameters()).device
    decoder = model.get_decoder()
    head_weight = model.get_output_embeddings().weight
    id1, id2 = token_attribution.answer_token_ids(tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    results = [None] * len(conditions)
    base_logprobs = None
    order = sorted(range(len(conditions)), key=lambda i: conditions[i]["layer"])
    assert conditions[0]["name"] == "baseline"
    order.remove(0)
    order.insert(0, 0)

    i = 0
    while i < len(order):
        idx = [order[i]]
        layer = conditions[order[i]]["layer"]
        while len(idx) < batch and i + len(idx) < len(order) \
                and conditions[order[i + len(idx)]]["layer"] == layer:
            idx.append(order[i + len(idx)])
        i += len(idx)

        rows = [conditions[j] for j in idx]
        ids = input_ids.expand(len(rows), -1)
        mask = attention_mask.expand(len(rows), -1)
        handle = None
        if layer > 0:
            u = torch.stack([torch.from_numpy(r["vector"]) for r in rows]).to(
                device=device, dtype=next(model.parameters()).dtype)
            dose = torch.tensor([r["dose"] for r in rows], device=device,
                                dtype=u.dtype)
            pos = None
            touched = [ids.shape[1]] * len(rows)
            if any(r.get("restriction", "full") != "full" for r in rows):
                assert meta is not None, "restricted conditions need the prompt meta"
                masks = [position_mask(meta, r.get("restriction", "full"),
                                        r["name"], ids.shape[1]) for r in rows]
                touched = [int(m.sum()) for m in masks]
                pos = torch.tensor(np.stack(masks), device=device, dtype=u.dtype)
            handle = decoder.layers[layer - 1].register_forward_hook(
                _projection_hook(u, dose, pos))
        try:
            with torch.no_grad():
                hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            if handle is not None:
                handle.remove()

        with torch.no_grad():
            last = hidden[:, -1].float()
            logits_last = last @ head_weight.T.float()
            logprobs = torch.log_softmax(logits_last, dim=-1)
            gaps = (logits_last[:, id1] - logits_last[:, id2]).tolist()
            nlls = _nll_per_row(hidden, head_weight, ids)
        for row_i, j in enumerate(idx):
            if j == 0:
                base_logprobs = logprobs[row_i]
                kl = 0.0
            else:
                assert base_logprobs is not None, "baseline must run first"
                kl = float(torch.sum(base_logprobs.exp()
                                     * (base_logprobs - logprobs[row_i])))
            results[j] = {"gap": gaps[row_i], "kl_answer": kl, "nll": nlls[row_i],
                          "n_touched": touched[row_i] if layer > 0 else ids.shape[1]}
    return results


def projection_rows(model, tokenizer, df, c, conditions, batch=4):
    """Yield one condition table per rendered prompt."""
    for _, row in df.iterrows():
        meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
        enc = tokenizer(meta["prompt"], return_tensors="pt")
        assert enc["input_ids"].shape[1] == len(meta["input_ids"])
        res = run_prompt(model, tokenizer, enc["input_ids"], enc["attention_mask"],
                         conditions, batch, meta)
        yield pd.DataFrame({
            "pair_uid": row["pair_uid"], "item": row["item"], "order": int(row["order"]),
            "name": [r["name"] for r in conditions],
            "layer": [r["layer"] for r in conditions],
            "dose": [r["dose"] for r in conditions],
            "restriction": [r.get("restriction", "full") for r in conditions],
            "gap": [x["gap"] for x in res],
            "kl_answer": [x["kl_answer"] for x in res],
            "nll": [x["nll"] for x in res],
            "n_touched": [x["n_touched"] for x in res]})


def main():
    import argparse

    import verdicts
    import wbconfig

    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True, help="npz of unit directions")
    parser.add_argument("--fit-layer", type=int, default=14)
    parser.add_argument("--control-layer", type=int, default=21)
    parser.add_argument("--doses", default="0.25,0.5,0.75,1")
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--restrictions", default="full",
                        help="comma list: full, not_last, role, not_role")
    parser.add_argument("--only-directions", default=None,
                        help="comma list of direction names to keep from the npz")
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-sampling", default="round_robin",
                        choices=("round_robin", "balanced"))
    parser.add_argument("--limit-seed", type=int, default=0)
    parser.add_argument("--min-reasoning-chars", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--chunk-prompts", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, tokenizer, _ = verdicts.load_model(args.model, None, c, dtype=torch.bfloat16)
    model.eval()

    pairs = wbpairs.load_pairs(c, args.split)
    if args.min_reasoning_chars:
        pairs = wbpairs.drop_answer_only_pairs(pairs, args.min_reasoning_chars)
    if args.limit_sampling == "balanced":
        df = token_manifold_record.balanced_limit_pairs(pairs, args.limit, args.limit_seed)
    else:
        df = token_manifold_record.limit_pairs(pairs, args.limit)
    df = token_attribution.select_shard(df, args.shard_index, args.shard_count)

    directions = load_directions(args.directions)
    if args.only_directions:
        keep = args.only_directions.split(",")
        assert set(keep) <= set(directions), sorted(directions)
        directions = {k: directions[k] for k in keep}
    doses = [float(s) for s in args.doses.split(",")]
    conditions = condition_table(directions, args.fit_layer, doses,
                                 args.control_layer, args.n_random, args.random_seed,
                                 tuple(args.restrictions.split(",")))
    print(f"{len(conditions)} conditions x {len(df)} prompts", flush=True)

    id1, id2 = token_attribution.answer_token_ids(tokenizer)
    with torch.no_grad():
        w = model.get_output_embeddings().weight
        readout = (w[id1] - w[id2]).float()
        readout = (readout / readout.norm()).cpu().numpy()
    for name, u in directions.items():
        print(f"cos({name}, unembedding difference) = {float(u @ readout):+.4f}",
              flush=True)

    done = token_attribution.completed_prompts(args.out) \
        if pathlib.Path(args.out).exists() else set()
    if done:
        keep = [(str(u), int(o)) not in done for u, o in zip(df["pair_uid"], df["order"])]
        print(f"resuming: {len(done)} done, {sum(keep)} to go", flush=True)
        df = df[keep]
    if df.empty:
        print("nothing left", flush=True)
        return
    token_attribution.write_occlusion_parts(
        projection_rows(model, tokenizer, df, c, conditions, args.batch),
        args.out, args.chunk_prompts)


if __name__ == "__main__":
    main()
