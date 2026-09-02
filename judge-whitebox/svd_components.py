"""Per-head/per-block SVD registry for Qwen3 attention and MLP weights.

Entries are raw sliced weight matrices (Q/K/V/O per-head rows or columns,
whole gate/up/down). Screening and ablation act on these directly, and the
factors are exactly writable back (W = U @ diag(S) @ V.T). QK/OV *product*
circuits (Q.T@K, O@V) are interpretation-only: RoPE and per-head q/k RMSNorm
sit inside those products and are not captured by decomposing the raw
projections separately.
"""
import torch

WEIGHT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "gate_proj.weight", "up_proj.weight", "down_proj.weight")


class Decomp:
    def __init__(self):
        self.entries = {}  # key -> (U, S, V)
        self.meta = {}  # key -> {layer, kind, head, kv_head}


def _layer_and_suffix(name):
    """"model.layers.{n}.X" -> (n, "X"), or None if name doesn't match."""
    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
        return None
    if not name.endswith(WEIGHT_SUFFIXES):
        return None
    return int(parts[2]), ".".join(parts[3:])


def _svd(mat):
    u, s, vh = torch.linalg.svd(mat.float().cpu(), full_matrices=False)
    return u, s, vh.T


def _add_head_entries(decomp, layer, kind, weight, n_heads, head_dim, by_columns, kv_head_of):
    for h in range(n_heads):
        lo, hi = h * head_dim, (h + 1) * head_dim
        sl = weight[:, lo:hi] if by_columns else weight[lo:hi, :]
        key = f"L{layer}.{kind}.h{h}"
        decomp.entries[key] = _svd(sl)
        decomp.meta[key] = {"layer": layer, "kind": kind, "head": h, "kv_head": kv_head_of(h)}


def decompose(state_dict, arch):
    """state_dict: HF Qwen3 weight names -> tensors. arch: {n_heads, n_kv_heads, head_dim}."""
    n_heads, n_kv_heads, head_dim = arch["n_heads"], arch["n_kv_heads"], arch["head_dim"]
    group_size = n_heads // n_kv_heads
    decomp = Decomp()

    for name, weight in state_dict.items():
        parsed = _layer_and_suffix(name)
        if parsed is None:
            continue
        layer, suffix = parsed

        if suffix == "self_attn.q_proj.weight":
            _add_head_entries(decomp, layer, "q", weight, n_heads, head_dim,
                               by_columns=False, kv_head_of=lambda h: h // group_size)
        elif suffix == "self_attn.o_proj.weight":
            _add_head_entries(decomp, layer, "o", weight, n_heads, head_dim,
                               by_columns=True, kv_head_of=lambda h: h // group_size)
        elif suffix == "self_attn.k_proj.weight":
            _add_head_entries(decomp, layer, "k", weight, n_kv_heads, head_dim,
                               by_columns=False, kv_head_of=lambda h: h)
        elif suffix == "self_attn.v_proj.weight":
            _add_head_entries(decomp, layer, "v", weight, n_kv_heads, head_dim,
                               by_columns=False, kv_head_of=lambda h: h)
        elif suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            kind = suffix.split(".")[1].split("_")[0]  # gate/up/down
            key = f"L{layer}.{kind}"
            decomp.entries[key] = _svd(weight)
            decomp.meta[key] = {"layer": layer, "kind": kind, "head": None, "kv_head": None}

    return decomp


def save(decomp, path):
    torch.save({"entries": decomp.entries, "meta": decomp.meta}, path)


def load(path):
    blob = torch.load(path)
    decomp = Decomp()
    decomp.entries = blob["entries"]
    decomp.meta = blob["meta"]
    return decomp
