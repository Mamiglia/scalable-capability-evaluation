"""Per-layer direction packs for multi-layer steering.

For each layer in interp_pod.mix.stage2_layers: refit the shared and
qwen-only BT approval directions, the capability ridge direction, and
v_pref_judge (qwen approval with capability projected out) on that
layer's last-token activations; bundle with feature stats, the layer's
median residual norm, and the seeded random controls. One pack file per
layer, ready for pod_mix --extra-packs.

Usage:
  uv run python scripts/analyses/judge_interp/prep_stage2_packs.py --staging <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from bt_probe import full_fit, verdict_diffs

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--staging", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    layers = cfg["interp_pod"]["mix"]["stage2_layers"]
    args.staging.mkdir(parents=True, exist_ok=True)

    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    pi = meta["poolings"].index("last_token")
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    index = {(t, i): k for k, (t, i) in enumerate(zip(elo["text_id"], elo["item"]))}

    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    cap = elo["model"].map(capability).to_numpy()
    cap = cap - pd.Series(cap).groupby(elo["item"]).transform("mean").to_numpy()

    for layer in layers:
        li = meta["layers"].index(layer)
        features = np.asarray(acts[:, li, pi, :], dtype=np.float32)
        median_norm = float(np.median(np.linalg.norm(features, axis=1)))

        rows, k1, k2 = verdict_diffs(cfg, features, index)
        model, mean, std, c = full_fit(rows, k1, k2, features, cfg["bt_probe"])
        qrows = rows[rows["judge"] == "qwen3-32b"].reset_index(drop=True)
        qk1 = np.array([index[(t, i)] for t, i in zip(qrows["text_id_slot1"], qrows["item"])])
        qk2 = np.array([index[(t, i)] for t, i in zip(qrows["text_id_slot2"], qrows["item"])])
        qmodel, _, _, qc = full_fit(qrows, qk1, qk2, features, cfg["bt_probe"])

        z = (features - mean) / std
        v_cap = Ridge(cfg["probes"]["alpha_grid"][-1]).fit(z, cap).coef_
        v_cap_unit = v_cap / np.linalg.norm(v_cap)
        w_qwen = qmodel.coef_[0]
        v_pref_judge = w_qwen - (v_cap_unit @ w_qwen) * v_cap_unit

        rng = np.random.default_rng(cfg["interp_pod"]["seed"])
        dim = features.shape[1]
        directions = dict(
            approval=model.coef_[0], approval_qwen=w_qwen, v_cap=v_cap,
            v_pref_judge=v_pref_judge,
            random_a=rng.standard_normal(dim), random_b=rng.standard_normal(dim))
        path = args.staging / f"pack_l{layer}.npz"
        np.savez(path, layer=layer, pooling="last_token", mean=mean, std=std,
                 median_norm=median_norm, prompt=cfg["record"]["prompt"],
                 names=list(directions), **{f"dir_{k}": v for k, v in directions.items()})
        print(f"layer {layer}: C={c}/{qc}, median norm {median_norm:.1f} -> {path}")


if __name__ == "__main__":
    main()
