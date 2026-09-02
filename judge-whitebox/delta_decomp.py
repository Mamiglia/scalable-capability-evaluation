"""LoRA delta decomposed in the base model's SVD basis.

For each registry key whose parent weight has an adapter delta: ΔW is sliced with the same
row/col convention as the registry entry (screen._slice_bounds), then rotated into the top-k
retained base components M = U[:, :k].T @ ΔW_slice @ V[:, :k], k = min(screen.store_top_k,
this key's SVD rank). Energy splits diag (same component reused) / off-diag (components mixed)
/ outside (energy the retained basis doesn't capture at all — truncated-out tail components or
directions foreign to this key's basis) — the three sum exactly to ‖ΔW_slice‖²_F, showing
whether a LoRA edit reuses directions the base model already carries.
"""
import argparse
import json
import pathlib

import pandas as pd
import safetensors.torch
import torch

import importance_report
import screen
import svd_components
import wbconfig
import wbfig

PEFT_PREFIX = "base_model.model."


def load_adapter_deltas(adapter_path):
    """adapter_path: peft dir with adapter_config.json + adapter_model.{safetensors,bin}.
    Never loads the base model. Returns hf_weight_name -> ΔW (float32) = (alpha/r) * B @ A.
    r is read per-matrix from the tensor shape (rank-agnostic), not from config["r"].
    """
    adapter_path = pathlib.Path(adapter_path)
    config = json.loads((adapter_path / "adapter_config.json").read_text())
    alpha = config["lora_alpha"]

    sft, bin_ = adapter_path / "adapter_model.safetensors", adapter_path / "adapter_model.bin"
    if sft.exists():
        weights = safetensors.torch.load_file(sft)
    elif bin_.exists():
        weights = torch.load(bin_, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no adapter_model.safetensors or .bin under {adapter_path}")

    deltas = {}
    for key, a in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        b = weights[prefix + ".lora_B.weight"]
        hf_name = prefix.removeprefix(PEFT_PREFIX) + ".weight"
        r = a.shape[0]
        deltas[hf_name] = (alpha / r) * (b.float() @ a.float())
    return deltas


def _hf_weight_name(meta):
    group = "self_attn" if meta["kind"] in ("q", "k", "v", "o") else "mlp"
    return f"model.layers.{meta['layer']}.{group}.{meta['kind']}_proj.weight"


def _top5(M):
    rank = M.shape[0]
    flat_abs, flat_signed = M.abs().flatten(), M.flatten()
    top_idx = torch.topk(flat_abs, min(5, flat_abs.numel())).indices
    return [(int(idx // rank), int(idx % rank), flat_signed[idx].item()) for idx in top_idx]


def delta_in_base_basis(decomp, deltas):
    """Per key whose parent matrix has an adapter delta: rotate the delta's registry-matching
    slice into the top-k retained (U, V) basis and split its energy diag / off-diag / outside.
    """
    store_top_k = wbconfig.cfg()["screen"]["store_top_k"]
    rows = []
    for key, (U, S, V) in decomp.entries.items():
        meta = decomp.meta[key]
        hf_name = _hf_weight_name(meta)
        if hf_name not in deltas:
            continue
        by_columns = meta["kind"] == "o"
        lo, hi = screen._slice_bounds(by_columns, meta["head"], U, V)
        delta = deltas[hf_name].float()
        delta_slice = delta[:, lo:hi] if by_columns else delta[lo:hi, :]

        rank = S.shape[0]
        k = min(store_top_k, rank)
        M = U[:, :k].float().T @ delta_slice @ V[:, :k].float()

        diag = torch.diagonal(M)
        energy_diag = (diag ** 2).sum().item()
        energy_offdiag = (M ** 2).sum().item() - energy_diag
        energy_total_delta = (delta_slice ** 2).sum().item()
        energy_outside = energy_total_delta - (energy_diag + energy_offdiag)

        rows.append({
            "key": key, "layer": meta["layer"], "kind": meta["kind"], "head": meta["head"],
            "rank": rank, "topk_k": k,
            "energy_diag": energy_diag, "energy_offdiag": energy_offdiag,
            "energy_outside": energy_outside, "energy_total_delta": energy_total_delta,
            "top5": _top5(M),
        })
    return pd.DataFrame(rows)


def fig_energy(df, tag):
    """Stacked bars per layer, energy fraction diag/off-diag/outside, one panel per matrix kind
    present in df. Fractions are of energy_total_delta, summed over heads within a layer."""
    kinds = [k for k in importance_report.KINDS if k in set(df["kind"])]
    agg = df.groupby(["kind", "layer"])[
        ["energy_diag", "energy_offdiag", "energy_outside", "energy_total_delta"]].sum()

    fig = wbfig.new_fig(wbfig.WIDE, 2.2 * len(kinds))
    for i, kind in enumerate(kinds):
        ax = fig.add_subplot(len(kinds), 1, i + 1)
        sub = agg.loc[kind].sort_index()
        total = sub["energy_total_delta"].clip(lower=1e-12)
        frac_diag = sub["energy_diag"] / total
        frac_offdiag = sub["energy_offdiag"] / total
        frac_outside = sub["energy_outside"] / total
        ax.bar(sub.index, frac_diag, color=wbfig.PALETTE[0], label="diag")
        ax.bar(sub.index, frac_offdiag, bottom=frac_diag, color=wbfig.PALETTE[1], label="off-diag")
        ax.bar(sub.index, frac_outside, bottom=frac_diag + frac_offdiag,
               color=wbfig.PALETTE[2], label="outside")
        ax.set_ylabel(kind)
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.legend(frameon=False, ncol=3)
        if i == len(kinds) - 1:
            ax.set_xlabel("layer")
    wbfig.save(fig, f"delta_{tag}_energy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    c = wbconfig.cfg()
    decomp = svd_components.load(args.decomp)
    deltas = load_adapter_deltas(args.adapter)
    df = delta_in_base_basis(decomp, deltas)

    out_path = pathlib.Path(c["paths"]["out"]) / f"delta_{args.tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    fig_energy(df, args.tag)

    print(f"n_keys={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()
