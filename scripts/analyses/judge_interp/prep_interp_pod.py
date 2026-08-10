"""Package directions and transcript selections for the pod run.

Selects transcripts by verdict-trained approval score (top, bottom,
mid), bundles every direction (approval shared/qwen, v_cap, v_pref,
v_pref orthogonalized, seeded random controls) with feature stats,
the probe layer, and the median raw activation norm there (steering
strengths are set as fractions of it). Writes a staging dir with
interp_pack.npz + selection.parquet ready to scp to the pod.

Usage:
  uv run python scripts/analyses/judge_interp/prep_interp_pod.py --staging <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def storage_root(cfg) -> Path:
    roots = [Path(r) for r in cfg["storage_roots"]]
    for root in roots:
        if root.is_dir():
            return root
    raise SystemExit(f"no storage root mounted; tried {roots}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--staging", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    sel_cfg = cfg["interp_pod"]
    args.staging.mkdir(parents=True, exist_ok=True)

    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    bt = np.load(out / "bt_direction.npz")
    capd = np.load(out / "capability_directions.npz")
    assert int(bt["layer"]) == int(capd["layer"])
    layer = int(bt["layer"])
    li = meta["layers"].index(layer)
    pi = meta["poolings"].index(str(bt["pooling"]))
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    features = np.asarray(acts[:, li, pi, :], dtype=np.float32)
    median_norm = float(np.median(np.linalg.norm(features, axis=1)))

    mean, std = bt["mean"], bt["std"]
    z = (features - mean) / std
    score = z @ bt["w"]

    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    order = np.argsort(score)
    n_top, n_bot, n_mid = (sel_cfg[k] for k in ("n_top", "n_bottom", "n_mid"))
    mid0 = len(order) // 2 - n_mid // 2
    picks = np.concatenate([order[-n_top:], order[:n_bot],
                            order[mid0:mid0 + n_mid]])
    roles = (["top"] * n_top + ["bottom"] * n_bot + ["mid"] * n_mid)
    chosen = elo.iloc[picks].assign(role=roles, approval_score=score[picks])

    inputs = pd.read_parquet(storage_root(cfg) / cfg["record"]["inputs"])
    selection = chosen.merge(inputs, on=["text_id", "item"], how="left",
                             suffixes=("", "_inputs"))
    assert selection["text"].notna().all(), "selection rows missing inputs"
    selection.to_parquet(args.staging / "selection.parquet", index=False)

    rng = np.random.default_rng(sel_cfg["seed"])
    dim = features.shape[1]
    v_cap_unit = capd["v_cap"] / np.linalg.norm(capd["v_cap"])
    directions = dict(
        approval=bt["w"], approval_qwen=bt["w_qwen"],
        v_cap=capd["v_cap"], v_pref=capd["v_pref"],
        v_pref_orth=capd["v_pref_orth"],
        v_pref_judge=bt["w_qwen"] - (v_cap_unit @ bt["w_qwen"]) * v_cap_unit,
        random_a=rng.standard_normal(dim), random_b=rng.standard_normal(dim))
    np.savez(args.staging / "interp_pack.npz",
             layer=layer, pooling=str(bt["pooling"]), mean=mean, std=std,
             median_norm=median_norm, prompt=cfg["record"]["prompt"],
             names=list(directions), **{f"dir_{k}": v for k, v in directions.items()})
    print(f"{len(selection)} transcripts "
          f"({n_top} top / {n_bot} bottom / {n_mid} mid), "
          f"layer {layer}, median residual norm {median_norm:.1f}")
    print(f"-> {args.staging}/interp_pack.npz, selection.parquet")


if __name__ == "__main__":
    main()
