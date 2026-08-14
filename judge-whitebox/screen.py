"""Importance screen: per-component activity and loss attribution vs a control corpus.

For a registry entry (U, S, V) hosted on nn.Linear y = x @ W.T:
- ROW-sliced entries (q/k/v, and MLP whole, by_columns=False): x is the full input;
  the entry's rows of W are a slice [lo:hi] of the output, so grad is taken on that
  output slice. lo/hi come from meta["head"] * component-count (U.shape[0]), or the
  full range when head is None (MLP whole).
- COLUMN-sliced entries (o, by_columns=True): the entry reads a slice [lo:hi] of the
  input and writes all outputs, so x is sliced and grad is taken on the full output.
activity[i]  = sum_tokens S[i] * |V[:,i]^T x|
attribution[i] = sum_tokens S[i] * (V[:,i]^T x) * (U[:,i]^T grad_y), signed and abs (per-prompt |sum| for attr_abs).

For the global top-k components (by activity), the screen additionally stores per_prompt_attr
(signed per-prompt sum of the attribution term above, unnormalised — see ScreenResult) and
per_prompt_proj (signed per-prompt token-mean of the raw V[:,i]^T x, no S scale, no gradient).
When the full [n_prompts, all_components] store fits screen.per_prompt_budget_mb, pass 1 itself
captures all components per prompt and the top-k columns are sliced out afterwards (single-pass,
see run_screen). Otherwise (e.g. an 8B full-split screen) pass 1 only accumulates totals, and a
second per-prompt pass (_per_prompt_matrices) re-runs forward+backward per prompt, hooking only
the top-k's keys.
"""
import argparse
import os
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import corpus
import svd_components
import wbconfig
import wbpairs
import wbprompts


class ScreenResult:
    def __init__(self, activity, attr_signed, attr_abs, n_tokens, n_prompts,
                 top_k_components, per_prompt_activity, prompt_tokens,
                 kept_row_idx, split, limit, set_name, model, adapter,
                 per_prompt_attr=None, per_prompt_proj=None, invert=False, pairs_path=None):
        self.activity = activity  # key -> float32 tensor [r], summed over tokens
        self.attr_signed = attr_signed  # key -> float32 tensor [r], summed over tokens
        self.attr_abs = attr_abs  # key -> float32 tensor [r], summed over prompts of |per-prompt sum|
        self.n_tokens = n_tokens
        self.n_prompts = n_prompts
        self.top_k_components = top_k_components  # list of (key, component_idx), by global activity
        # mean per token (row p / prompt_tokens[p]); recover the raw sum as row * prompt_tokens[p]
        self.per_prompt_activity = per_prompt_activity  # float32 [n_prompts, len(top_k_components)]
        self.prompt_tokens = prompt_tokens  # list[int], aligned with per_prompt_activity rows
        # signed, raw per-prompt sum (NOT length-normalised, unlike per_prompt_activity above):
        # attribution is already a per-prompt sum by construction (it's the same pre-abs quantity
        # summed over prompts to make attr_abs), so there's no token count to divide out.
        self.per_prompt_attr = per_prompt_attr  # float32 [n_prompts, len(top_k_components)] or None
        # signed token-MEAN of the raw (unscaled by S) input projection V[:,i]^T x_t; x-side only,
        # no gradient — comes from the same per-prompt pass as per_prompt_attr (see _per_prompt_matrices).
        self.per_prompt_proj = per_prompt_proj  # float32 [n_prompts, len(top_k_components)] or None
        self.invert = invert  # whether this (judging-only) screen ran on inverted verdict prompts
        # position (0-based) of each kept row in `batches` — rows are dropped when over
        # batch_tokens, so this is what lets a consumer recover the exact prompt <-> source-row
        # alignment. Always populated (batch positions), but pairs_df.iloc[kept_row_idx] only
        # means anything for set_name == "judging" (row-per-pair); for "control" a single pairs-df
        # row fans out into several batch entries (see corpus.build_control), so these positions
        # do NOT index into pairs_df at all. Consumers must check set_name, not just look for None.
        self.kept_row_idx = kept_row_idx
        self.split = split  # split arg load_pairs was called with, or None
        self.limit = limit  # limit arg load_pairs was called with, or None
        self.set_name = set_name  # "judging" | "control" | None — which --set this screen ran
        self.model = model  # model name key (models.hf_ids/hpc_dir), or None — identity check vs a paired screen
        self.adapter = adapter  # adapter path, or None (base model)
        self.pairs_path = pairs_path  # resolved paths.pairs (WB_PAIRS-aware) this screen's batches came
        # from, or None (legacy) — two files can share a split NAME, so consumers pinning down which
        # pairs file kept_row_idx indexes into must check this, not just split/limit.

    def save(self, path):
        torch.save(vars(self), path)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, adapter, c):
    model_path = wbconfig.resolve_model(model_name, c)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    device = pick_device()
    model.to(device)
    model.eval()
    model.requires_grad_(False)  # screening never trains; backward flows via inputs_embeds instead
    return model, tokenizer, device


def qwen_entry_map(model, decomp):
    """key -> (nn.Linear, by_columns) using the Qwen3 HF module layout (model.model.layers[n].*)."""
    layers = model.model.layers
    entry_map = {}
    for key, meta in decomp.meta.items():
        layer = layers[meta["layer"]]
        kind = meta["kind"]
        module = getattr(layer.self_attn if kind in ("q", "k", "v", "o") else layer.mlp, f"{kind}_proj")
        entry_map[key] = (module, kind == "o")
    return entry_map


def _slice_bounds(by_columns, head, U, V):
    """lo/hi of this entry's slice: output columns (row-sliced) or input columns (column-sliced).
    width comes from U/V shape directly (not from arch config) — see qwen_entry_map/decomp meta.
    """
    width = V.shape[0] if by_columns else U.shape[0]
    lo = 0 if head is None else head * width
    return lo, lo + width


def _project_key(U, S, V, by_columns, head, x, g=None):
    """x: [T, in_full] float32 on x's device. g: [T, out_full] float32 or None.
    Moves U/S/V to x's device/dtype per call; hoist once per run if this is ever the bottleneck.
    Returns (activity[r], x_proj[T,r], attr[T,r] or None): x_proj is the raw signed input
    projection V[:,i]^T x_t (no S scale) — activity and attr are both built from it (|.|*S,
    resp. *S*g_proj), and per_prompt_proj reuses this same raw value directly. attr, summed
    over T, is signed attribution.
    """
    U, S, V = U.to(x.device, torch.float32), S.to(x.device, torch.float32), V.to(x.device, torch.float32)
    if g is not None:
        g = g.to(x.device, torch.float32)
    lo, hi = _slice_bounds(by_columns, head, U, V)
    if by_columns:
        x_proj = x[:, lo:hi] @ V
        g_proj = (g @ U) if g is not None else None
    else:
        x_proj = x @ V
        g_proj = (g[:, lo:hi] @ U) if g is not None else None
    activity = (S * x_proj.abs()).sum(dim=0)
    attr = (S * x_proj * g_proj) if g_proj is not None else None
    return activity, x_proj, attr


def _zero_stats(S):
    z = torch.zeros_like(S)
    return z, z.clone(), z.clone()


def _attach_hooks(decomp, entry_map, x_cache, stats, proj_capture=None):
    """Forward hook per hosted Linear caches its input x. If stats is not None, also puts a
    tensor .register_hook on the output that projects THIS key as soon as its grad arrives,
    writes the result into stats, and drops both x and the grad right there — no full grad_y (or
    x, past its own use) is retained for the life of the backward pass. That streaming is what
    keeps an 8B model at multi-thousand-token batches from OOMing on (n_layers x d_model x T)
    grad tensors plus (unused) parameter grads; the latter is avoided by freezing params and
    differentiating from an inputs_embeds leaf instead (see _logits_with_grad/load_model).
    Prefers output.register_hook over module.register_full_backward_hook, which is flaky for
    modules that fire multiple hooks (shared across heads) and has had reliability issues on MPS.
    Pass stats=None for a forward-only (activity-only) pass, which skips the backward hook.
    proj_capture: optional dict the backward hook also fills with key -> x_proj.mean(dim=0) (the
    raw, unscaled-by-S signed projection, token-averaged) — ignored when stats is None, since
    that path never reaches the backward hook this comes from. Used by the per-prompt pass
    (_per_prompt_matrices) to get per_prompt_proj for free out of the same backward it already
    runs for per_prompt_attr.
    """
    handles = []
    for key, (module, by_columns) in entry_map.items():
        def fwd(mod, inputs, output, key=key, by_columns=by_columns):
            x_cache[key] = inputs[0].detach()
            if stats is None:
                return

            def bwd(grad, key=key, by_columns=by_columns):
                x = x_cache.pop(key)
                x2, g2 = x.reshape(-1, x.shape[-1]).float(), grad.detach().reshape(-1, grad.shape[-1]).float()
                U, S, V = decomp.entries[key]
                head = decomp.meta[key].get("head")
                activity, x_proj, attr = _project_key(U, S, V, by_columns, head, x2, g2)
                stats[key] = (activity.cpu(), attr.sum(dim=0).cpu(), attr.sum(dim=0).abs().cpu())
                if proj_capture is not None:
                    proj_capture[key] = x_proj.mean(dim=0).cpu()

            output.register_hook(bwd)
        handles.append(module.register_forward_hook(fwd))
    return handles


def screen_one(model, decomp, entry_map, forward_fn, proj_capture=None):
    """One backward pass: forward_fn() runs model and returns a scalar loss tensor.
    Returns key -> (activity[r], attr_signed[r], attr_abs[r]) for this single prompt/batch.
    Generic over model/decomp/entry_map so a toy nn.Module + hand-built entry_map can bypass
    qwen_entry_map entirely (used by the finite-difference test).
    proj_capture: optional dict, filled key -> mean signed x-projection (see _attach_hooks);
    unused by the pass-1 aggregate loop, used by _per_prompt_matrices for per_prompt_proj.
    """
    x_cache, stats = {}, {}
    handles = _attach_hooks(decomp, entry_map, x_cache, stats, proj_capture=proj_capture)
    try:
        model.zero_grad(set_to_none=True)
        forward_fn().backward()
    finally:
        for h in handles:
            h.remove()
    for key in entry_map:
        stats.setdefault(key, _zero_stats(decomp.entries[key][1]))  # loss didn't touch this key
    return stats


def _resolve_text(entry, loss_kind, tokenizer):
    """entry is a rendered prompt string for "gap", or a corpus dict for "nll"."""
    if loss_kind == "gap":
        return entry
    if "messages" in entry:
        return tokenizer.apply_chat_template(entry["messages"], tokenize=False, add_generation_prompt=False)
    return entry["text"]


def _logits_with_grad(model, ids):
    """Forward with grad flowing from a fresh inputs_embeds leaf, not from model parameters
    (which load_model freezes). Gradient w.r.t. every intermediate activation is unaffected —
    the chain rule for grad_x doesn't need W.requires_grad — but autograd no longer accumulates
    a .grad per parameter, which is the other half of the OOM this screen would otherwise hit.
    """
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    return model(inputs_embeds=embeds).logits[0].float()


def _loss(logits, ids, loss_kind, tokenizer):
    """logits: [T, vocab] float32. ids: [T] token ids of the same sequence."""
    if loss_kind == "gap":
        id1, id2 = [tokenizer.encode(t, add_special_tokens=False) for t in wbprompts.ANSWER_TOKENS]
        assert len(id1) == 1 and len(id2) == 1, f"answer tokens must be single-token: {id1, id2}"
        return logits[-1, id1[0]] - logits[-1, id2[0]]
    return torch.nn.functional.cross_entropy(logits[:-1], ids[1:])


def _select_top_k(activity_sum, k):
    flat = [(act[i].item(), key, i) for key, act in activity_sum.items() for i in range(act.shape[0])]
    flat.sort(key=lambda t: -t[0])
    return [(key, i) for _, key, i in flat[:k]]


def _per_prompt_matrices(model, decomp, entry_map, kept_ids, prompt_tokens, top_components,
                          loss_kind, tokenizer):
    """Second-pass fallback, one prompt at a time, hooking only the (few) matrices holding a top
    component. Used by run_screen only when the full [n_prompts, all_components] store would
    exceed screen.per_prompt_budget_mb (e.g. an 8B full-split screen) — under budget, run_screen
    captures all components per prompt during pass 1 itself and never calls this at all.

    Design choice: per_prompt_attr needs gradients, so this pass is no longer forward-only —
    it re-runs one forward+backward per prompt (same as pass 1 already pays per prompt), but
    with hooks restricted to only the handful of keys holding a top component, which keeps the
    added cost the same order as the forward-only pass this replaces. per_prompt_proj doesn't
    need gradients on its own, but comes for free out of the same backward hook (it's the x-side
    factor _project_key already forms en route to attribution), so there's no separate no-grad
    pass for it — see proj_capture on _attach_hooks/screen_one.

    Returns (per_prompt_activity, per_prompt_attr, per_prompt_proj), each [n_prompts, k]:
      activity: mean per token (unchanged meaning — multiply by prompt_tokens[p] for the raw sum).
      attr: signed, raw per-prompt sum, NOT length-normalised (attribution is already a
        per-prompt sum by construction; see ScreenResult.per_prompt_attr).
      proj: signed token-mean of the raw (unscaled by S) projection V[:,i]^T x_t.
    """
    if not top_components:
        z = torch.zeros(len(kept_ids), 0)
        return z, z, z
    needed = sorted({key for key, _ in top_components})
    reduced_map = {k: entry_map[k] for k in needed}
    idx_by_key = {}
    for j, (key, i) in enumerate(top_components):
        idx_by_key.setdefault(key, []).append((j, i))

    n = len(top_components)
    act_mat = torch.zeros(len(kept_ids), n)
    attr_mat = torch.zeros(len(kept_ids), n)
    proj_mat = torch.zeros(len(kept_ids), n)
    for p, ids in enumerate(kept_ids):
        proj_capture = {}

        def forward_fn(ids=ids):
            return _loss(_logits_with_grad(model, ids), ids[0], loss_kind, tokenizer)

        stats = screen_one(model, decomp, reduced_map, forward_fn, proj_capture=proj_capture)
        for key, pairs in idx_by_key.items():
            activity, attr_signed, _ = stats[key]
            proj = proj_capture.get(key, torch.zeros_like(activity))
            for j, i in pairs:
                act_mat[p, j] = activity[i].item() / prompt_tokens[p]
                attr_mat[p, j] = attr_signed[i].item()
                proj_mat[p, j] = proj[i].item()
    return act_mat, attr_mat, proj_mat


def run_screen(model, tokenizer, decomp, batches, loss_kind, c, split=None, limit=None, set_name=None,
                model_name=None, adapter=None, invert=False):
    """split/limit: the wbpairs.load_pairs args batches was built from, if any — stored on the
    result so a downstream consumer (e.g. case_profile.py) can reconstruct the exact pairs df.
    set_name: "judging" or "control", for consumers that only make sense on one of them.
    model_name/adapter: the --model/--adapter this screen ran with, stored on the result so a
    judging/control pair can be checked for a matching model+adapter (importance_report.merge_table).
    invert: whether `batches` were rendered with the inverted (pick-weaker) verdict prompt;
    judging-only, stored on the result as-is (see main()'s CLI-level assert).
    """
    entry_map = qwen_entry_map(model, decomp)
    device = next(model.parameters()).device
    max_tokens = c["screen"]["batch_tokens"]

    activity_sum = {k: torch.zeros_like(s) for k, (_, s, _) in decomp.entries.items()}
    attr_signed_sum = {k: torch.zeros_like(v) for k, v in activity_sum.items()}
    attr_abs_sum = {k: torch.zeros_like(v) for k, v in activity_sum.items()}

    # component order = concatenation over sorted keys, so a (key, i) pair from _select_top_k
    # maps to a single column col_offset[key] + i in the full per-prompt store below.
    sorted_keys = sorted(activity_sum)
    col_offset, off = {}, 0
    for k in sorted_keys:
        col_offset[k] = off
        off += activity_sum[k].shape[0]
    n_components_total = off

    # n_prompts here is the pre-skip candidate count (len(batches)) — the actual kept count is
    # only known after the loop, and this decision has to be made before it starts. Using the
    # upper bound only ever overestimates, i.e. only ever pushes toward the safe (two-pass) side.
    budget_mb = c["screen"]["per_prompt_budget_mb"]
    est_bytes = len(batches) * n_components_total * 4 * 3
    single_pass = est_bytes <= budget_mb * 1024 * 1024
    print(f"per-prompt store estimate: {len(batches)} prompts x {n_components_total} components "
          f"x 3 matrices x 4B = {est_bytes / 1e6:.1f}MB (budget {budget_mb}MB) "
          f"-> {'single' if single_pass else 'two'}-pass")

    if single_pass:
        full_act = torch.zeros(len(batches), n_components_total)
        full_attr = torch.zeros(len(batches), n_components_total)
        full_proj = torch.zeros(len(batches), n_components_total)

    kept_ids, prompt_tokens, kept_row_idx, n_skipped = [], [], [], 0
    for i, entry in enumerate(batches):
        text = _resolve_text(entry, loss_kind, tokenizer)
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[1] > max_tokens:
            n_skipped += 1
            continue
        ids = ids.to(device)

        def forward_fn(ids=ids):
            return _loss(_logits_with_grad(model, ids), ids[0], loss_kind, tokenizer)

        proj_capture = {} if single_pass else None
        stats = screen_one(model, decomp, entry_map, forward_fn, proj_capture=proj_capture)
        row = len(kept_ids)
        for key, (a, asig, aabs) in stats.items():
            activity_sum[key] += a
            attr_signed_sum[key] += asig
            attr_abs_sum[key] += aabs
            if single_pass:
                lo, hi = col_offset[key], col_offset[key] + a.shape[0]
                full_act[row, lo:hi] = a  # raw (un-normalised) sum; normalised by token count below
                full_attr[row, lo:hi] = asig
                full_proj[row, lo:hi] = proj_capture.get(key, torch.zeros_like(a))
        kept_ids.append(ids)
        prompt_tokens.append(ids.shape[1])
        kept_row_idx.append(i)

    print(f"skipped {n_skipped} items over {max_tokens} tokens; kept {len(kept_ids)}")
    n_tokens = sum(prompt_tokens)

    top_components = _select_top_k(activity_sum, c["screen"]["store_top_k"])
    if single_pass:
        n_kept = len(kept_ids)
        if not top_components:
            per_prompt_activity = per_prompt_attr = per_prompt_proj = torch.zeros(n_kept, 0)
        else:
            cols = [col_offset[key] + i for key, i in top_components]
            tokens_t = torch.tensor(prompt_tokens, dtype=torch.float32).unsqueeze(1)
            per_prompt_activity = full_act[:n_kept][:, cols] / tokens_t
            per_prompt_attr = full_attr[:n_kept][:, cols]
            per_prompt_proj = full_proj[:n_kept][:, cols]
    else:
        per_prompt_activity, per_prompt_attr, per_prompt_proj = _per_prompt_matrices(
            model, decomp, entry_map, kept_ids, prompt_tokens, top_components, loss_kind, tokenizer)

    return ScreenResult(activity_sum, attr_signed_sum, attr_abs_sum, n_tokens, len(kept_ids),
                         top_components, per_prompt_activity, prompt_tokens,
                         kept_row_idx, split, limit, set_name, model_name, adapter,
                         per_prompt_attr=per_prompt_attr, per_prompt_proj=per_prompt_proj, invert=invert,
                         pairs_path=c.get("paths", {}).get("pairs"))


def _try_load_decomp(path, wait_s=0.5):
    """Screen array tasks share this cache file, so a reader can race a writer's os.replace and
    find the file gone (or briefly unreadable) right after the caller's exists() check. One retry
    after a short wait; if it still fails, return None so the caller rebuilds+re-saves instead of
    crashing the job."""
    for attempt in (1, 2):
        try:
            return torch.load(path)
        except (FileNotFoundError, EOFError, RuntimeError):
            if attempt == 2:
                return None
            time.sleep(wait_s)


def _load_or_build_decomp(model, decomp_path, model_name, adapter):
    """Cache key is (model_name, adapter): a merged-adapter model's weights don't match the base
    model's SVD factors, so a stale cache must fail loudly rather than silently score garbage.
    """
    if decomp_path and pathlib.Path(decomp_path).exists():
        blob = _try_load_decomp(decomp_path)
        if blob is not None:
            assert blob.get("model") == model_name and blob.get("adapter") == adapter, (
                f"decomp cache identity mismatch at {decomp_path}: cached "
                f"model={blob.get('model')!r} adapter={blob.get('adapter')!r}, requested "
                f"model={model_name!r} adapter={adapter!r}. Delete the cache or use a different path.")
            decomp = svd_components.Decomp()
            decomp.entries, decomp.meta = blob["entries"], blob["meta"]
            return decomp
        # file vanished/unreadable across the retry window (raced a concurrent writer) — rebuild below
    arch = {"n_heads": model.config.num_attention_heads,
            "n_kv_heads": model.config.num_key_value_heads,
            "head_dim": getattr(model.config, "head_dim",
                                 model.config.hidden_size // model.config.num_attention_heads)}
    decomp = svd_components.decompose(model.state_dict(), arch)
    if decomp_path:
        path = pathlib.Path(decomp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        torch.save({"entries": decomp.entries, "meta": decomp.meta,
                    "model": model_name, "adapter": adapter}, tmp)
        os.replace(tmp, path)  # atomic: array tasks sharing this cache never see a partial write
    return decomp


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name key into models.hf_ids")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    parser.add_argument("--set", required=True, choices=["judging", "control"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decomp", default=None, help="SVD cache path: load if present, else compute+save")
    parser.add_argument("--invert", action="store_true",
                         help="judging set only: render the inverted (pick-weaker) verdict prompt")
    args = parser.parse_args()
    assert not (args.invert and args.set == "control"), "--invert is judging-set only"

    model, tokenizer, device = load_model(args.model, args.adapter, c)
    decomp = _load_or_build_decomp(model, args.decomp, args.model, args.adapter)

    split = c["verdicts"]["eval_split"]
    df = wbpairs.load_pairs(c, split=split, limit=args.limit)
    if args.set == "judging":
        batches = [wbprompts.render_verdict_prompt(row, tokenizer, invert=args.invert)
                   for _, row in df.iterrows()]
        loss_kind = "gap"
    else:
        batches = corpus.build_control(df)
        loss_kind = "nll"

    # kept_row_idx only lines up with df rows for judging (row-per-pair); control fans one df
    # row out into several corpus entries, so its positions aren't pairs-df row indices.
    run_split, run_limit = (split, args.limit) if args.set == "judging" else (None, None)
    result = run_screen(model, tokenizer, decomp, batches, loss_kind, c,
                         split=run_split, limit=run_limit, set_name=args.set,
                         model_name=args.model, adapter=args.adapter, invert=args.invert)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"n_prompts={result.n_prompts} n_tokens={result.n_tokens} -> {out_path}")


if __name__ == "__main__":
    main()
