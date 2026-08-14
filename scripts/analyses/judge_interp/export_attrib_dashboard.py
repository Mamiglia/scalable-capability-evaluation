"""Export attribution-dashboard data from the ML token maps (local, no model).

Reads the per-transcript maps from pod_token_maps_ml.py and writes the compact
JSON the sentences_ml viewer loads:
  sentences.json    one row per sentence, occlusion delta on every direction,
                    raw and length-adjusted (residual of delta on n_chars).
  transcripts.json  per-transcript role/elo/item, last-token scores, and the
                    per-token projections (within the transcript span) for the
                    token-shaded transcript view.
  meta.json         directions, their layers, counts, elo range.

Occlusion sign (from the maps): delta_last = full_last_score - score_without.
Positive => the sentence drives that direction UP; negative => DOWN.

Usage:
  uv run python scripts/analyses/judge_interp/export_attrib_dashboard.py \
      --maps judge_results/judge_interp/attrib_ml_260807/token_maps \
      --out interactive_viewer/data_ml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NAMES5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]
# Each direction was fit at its own layer (design status log 2026-08-07).
LAYERS = {"v_cap": 26, "v_pref_gen": 26, "v_pref_gen_orth": 26,
          "v_pref_spec": 50, "v_pref_spec_orth": 50}


def length_resid(delta, n_chars):
    """Occlusion delta with the length trend removed (global linear fit).

    Longer sentences mechanically give bigger |delta|; the residual of a
    delta~n_chars fit is the length-adjusted driver strength.
    """
    x, y = np.asarray(n_chars, float), np.asarray(delta, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return y
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", type=Path,
                    default=ROOT / "judge_results/judge_interp/attrib_ml_260807/token_maps")
    ap.add_argument("--out", type=Path, default=ROOT / "interactive_viewer/data_ml")
    ap.add_argument("--min-chars", type=int, default=25,
                    help="viewer drops shorter fragments from the driver ranking")
    args = ap.parse_args()

    paths = sorted(args.maps.glob("*.json"))
    if not paths:
        raise SystemExit(f"no token maps under {args.maps}")

    # author model per (text_id, item) — the maps only carry an empty role
    elo = pd.read_csv(ROOT / "judge_results/judge_interp/elo.csv")
    author = {(str(t), str(i)): str(m)
              for t, i, m in zip(elo["text_id"], elo["item"], elo["model"])}

    transcripts, rows = {}, []
    for p in paths:
        d = json.loads(p.read_text())
        names = d.get("directions", NAMES5)
        key = f"{d['text_id']}_{d['item']}"
        lo, hi = d.get("transcript_span", [0, len(d.get("tokens", []))])
        tp = d.get("token_projections", {})
        transcripts[key] = dict(
            text_id=d["text_id"], item=d["item"],
            model=author.get((str(d["text_id"]), str(d["item"])), d.get("role") or "?"),
            role=(d.get("role") or "?"), elo=round(float(d.get("elo", 0.0)), 3),
            last={n: round(float(d.get("last_token_scores", {}).get(n, 0.0)), 4)
                  for n in names},
            n_sentences=len(d.get("occlusion", [])),
            tokens=d.get("tokens", [])[lo:hi],
            token_proj={n: [round(float(v), 3) for v in tp[n][lo:hi]] for n in tp})
        for pos, occ in enumerate(d.get("occlusion", [])):
            rec = dict(key=key, pos=pos, sentence=occ["sentence"],
                       n_chars=occ.get("n_chars", len(occ["sentence"])))
            for n in names:
                rec[f"{n}_raw"] = round(float(occ["delta_last"][n]), 4)
            rows.append(rec)

    nchars = [r["n_chars"] for r in rows]
    for n in NAMES5:                       # length-adjusted delta, per direction
        res = length_resid([r[f"{n}_raw"] for r in rows], nchars)
        for r, v in zip(rows, res):
            r[f"{n}_len"] = round(float(v), 4)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sentences.json").write_text(json.dumps(rows))
    (args.out / "transcripts.json").write_text(json.dumps(transcripts))
    elos = [t["elo"] for t in transcripts.values()]
    (args.out / "meta.json").write_text(json.dumps(dict(
        directions=NAMES5, layers=LAYERS, min_chars=args.min_chars,
        n_transcripts=len(transcripts), n_sentences=len(rows),
        elo_min=round(min(elos), 3), elo_max=round(max(elos), 3))))
    print(f"{len(rows)} sentences, {len(transcripts)} transcripts -> {args.out}")


if __name__ == "__main__":
    main()
