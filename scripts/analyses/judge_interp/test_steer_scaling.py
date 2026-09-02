"""Self-test for the steering scaling math (CPU-only, no model).

Loads the multilayer pack and, for a synthetic hidden state, checks that the
add-vectors from pod_judge_steer.steer_vectors move the standardized projection
z*v_hat by exactly alpha*g (single) and alpha*g_block (block). This is the
correctness gate for the activation-space scaling.

  uv run python test_steer_scaling.py [--pack path]
"""

import argparse
from pathlib import Path

import numpy as np

from pod_judge_steer import NAMES5, _unit, steer_vectors

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACK = ROOT / "judge_results/judge_interp/directions_multilayer.npz"
TOL = 1e-5


def check(pack):
    rng = np.random.default_rng(0)
    worst = 0.0
    for name in NAMES5:
        for alpha in (0.05, 0.5, -1.0):
            # ---- single ----
            std = np.asarray(pack[f"sl_{name}_std"], dtype=np.float64)
            mean = np.asarray(pack[f"sl_{name}_mean"], dtype=np.float64)
            vhat = _unit(pack[f"sl_{name}_dir"])
            g = float(pack[f"sl_{name}_g"])
            L = int(pack[f"sl_{name}_layer"])
            h = rng.standard_normal(std.shape[0])
            dh = steer_vectors(pack, name, "single", alpha)[L]
            dproj = float(((h + dh - mean) / std) @ vhat - ((h - mean) / std) @ vhat)
            err = abs(dproj - alpha * g)
            worst = max(worst, err)
            assert err < TOL, (name, "single", alpha, dproj, alpha * g, err)

            # ---- block ----
            layers = pack[f"blk_{name}_layers"]
            stdb = np.asarray(pack[f"blk_{name}_std"], dtype=np.float64)   # (5, 5120)
            meanb = np.asarray(pack[f"blk_{name}_mean"], dtype=np.float64)
            vhatb = _unit(np.asarray(pack[f"blk_{name}_dir"]).ravel()).reshape(stdb.shape)
            gb = float(pack[f"blk_{name}_g"])
            H = rng.standard_normal(stdb.shape)
            vecs = steer_vectors(pack, name, "block", alpha)
            dproj = 0.0
            for j, Lj in enumerate(layers):
                z0 = (H[j] - meanb[j]) / stdb[j]
                z1 = (H[j] + vecs[int(Lj)] - meanb[j]) / stdb[j]
                dproj += float(z1 @ vhatb[j] - z0 @ vhatb[j])
            err = abs(dproj - alpha * gb)
            worst = max(worst, err)
            assert err < TOL, (name, "block", alpha, dproj, alpha * gb, err)
            if name == "v_pref_gen" and alpha == 0.5:
                print(f"{name:14s} single alpha=0.5 g={g:+.4f}  "
                      f"Δproj={((h + dh - mean) / std) @ vhat - ((h - mean) / std) @ vhat:+.6f} "
                      f"target={alpha * g:+.6f}")
                print(f"{name:14s} block  alpha=0.5 g_block={gb:+.4f}  "
                      f"Δproj={dproj:+.6f} target={alpha * gb:+.6f}")
    print(f"OK: all Δproj match alpha*g within {TOL} (worst |err| = {worst:.2e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    args = ap.parse_args()
    check(dict(np.load(args.pack, allow_pickle=True)))


if __name__ == "__main__":
    main()
