"""Pull recorded activations off storage into one local array.

Reads every per-entity .npy under the storage acts dir and writes, to
out_dir: acts_<judge>.npy of shape (n_entities, n_layers, 2, hidden)
fp16, row-aligned to elo.csv, plus acts_<judge>.meta.json (recorder
meta + row-order note). Probes memory-map the array and slice out one
(layer, pooling) matrix at a time. Stops if any elo entity lacks a
file. The per-entity files on storage are the archive and stay as they
are.

Usage:
  uv run python scripts/analyses/judge_interp/consolidate_acts.py
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
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    acts_dir = storage_root(cfg) / cfg["record"]["acts_dir"]
    name = f"acts_{acts_dir.name}"

    elo = pd.read_csv(out / "elo.csv")
    entities = list(zip(elo["text_id"], elo["item"]))
    missing = [f"{t}_{i}" for t, i in entities
               if not (acts_dir / f"{t}_{i}.npy").exists()]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(entities)} entities lack "
                         f"activations, e.g. {missing[:3]}")

    first = np.load(acts_dir / f"{entities[0][0]}_{entities[0][1]}.npy")
    acts = np.empty((len(entities),) + first.shape, dtype=first.dtype)
    for k, (t, i) in enumerate(entities):
        acts[k] = np.load(acts_dir / f"{t}_{i}.npy")
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(entities)}", flush=True)

    np.save(out / f"{name}.npy", acts)
    meta = json.loads((acts_dir / "meta.json").read_text())
    meta["row_order"] = "elo.csv row order"
    meta["shape"] = list(acts.shape)
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{acts.shape} {acts.dtype} row-aligned to elo.csv "
          f"-> {out / f'{name}.npy'}")


if __name__ == "__main__":
    main()
