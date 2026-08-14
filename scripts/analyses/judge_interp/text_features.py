"""Which properties of a transcript track ability, and which only track taste.

Scores every recorded transcript on plain surface features, then relates
them to three things: the author's real benchmark strength, the judges'
consensus preference, and the preference that survives after real
strength is regressed out. Features that predict the last one are what
judges reward without it being ability.

Coefficients are standardized (feature and target z-scored), fit jointly
so correlated features - length above all - do not each take credit for
the same effect.

Writes out_dir/text_features.csv.

Usage:
  uv run python scripts/analyses/judge_interp/text_features.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[3]


def features(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    words = text.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    n_words = max(len(words), 1)
    def per_1k(n):
        return 1000 * n / n_words
    return {
        "length (words)": len(words),
        "words per sentence": n_words / max(len(sentences), 1),
        "equations": per_1k(len(re.findall(r"\$|\\frac|\\times|\\approx|=", text))),
        "numbers": per_1k(len(re.findall(r"\d", text))),
        "headings": per_1k(len(re.findall(r"^\s*(#{1,4}|\*\*[^*]+\*\*\s*$)", text, re.M))),
        "bullets": per_1k(len(re.findall(r"^\s*[-*+]\s", text, re.M))),
        "numbered steps": per_1k(len(re.findall(r"^\s*\d+[.)]\s", text, re.M))),
        "bold markup": per_1k(len(re.findall(r"\*\*", text))),
        "option mentions": per_1k(len(re.findall(r"\b(option|choice)\s*[A-D]\b|\b[A-D]\)", text, re.I))),
        "hedging": per_1k(len(re.findall(
            r"\b(might|maybe|perhaps|possibly|probably|seems|appears|likely|unclear|assume)\b",
            text, re.I))),
        "certainty": per_1k(len(re.findall(
            r"\b(clearly|obviously|definitely|certainly|must be|therefore|thus|hence)\b",
            text, re.I))),
        "self-correction": per_1k(len(re.findall(
            r"\b(wait|actually|let me reconsider|on second thought|correction|oops|hmm)\b",
            text, re.I))),
        "checking work": per_1k(len(re.findall(
            r"\b(verify|check|double-check|confirm|sanity)\b", text, re.I))),
        "units": per_1k(len(re.findall(
            r"\b(eV|nm|mol|kg|Hz|K|J|cm|mm|GeV|MeV|Å)\b", text))),
        "citations of theory": per_1k(len(re.findall(
            r"\b(theorem|principle|law|equation|formula|rule)\b", text, re.I))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    elo = pd.read_csv(out / "elo.csv")
    texts = pd.read_parquet(ROOT / cfg["texts_parquet"]).set_index("text_id")["text"]
    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]

    df = pd.DataFrame([features(texts[t]) for t in elo["text_id"]], index=elo.index)
    # Item-centre everything: the comparison judges make is within an item.
    item = elo["item"].to_numpy()
    def centre(v):
        return v - pd.Series(v).groupby(item).transform("mean").to_numpy()

    cap = centre(elo["model"].map(capability).to_numpy())
    pref = centre(elo["elo"].to_numpy())
    slope = float(np.polyfit(cap, pref, 1)[0])
    beyond = pref - slope * cap            # preference not explained by real ability

    x = np.column_stack([centre(df[c].to_numpy()) for c in df.columns])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    targets = {"real ability": cap, "judge preference": pref,
               "preference beyond ability": beyond}
    print(f"{len(elo)} transcripts, {x.shape[1]} features; "
          f"preference-vs-ability slope {slope:.3f}")

    rows = {}
    for name, y in targets.items():
        yz = (y - y.mean()) / (y.std() + 1e-9)
        model = RidgeCV(alphas=cfg["probes"]["alpha_grid"]).fit(x, yz)
        rows[name] = model.coef_
        print(f"  {name:26s} R2 {model.score(x, yz):.3f}")

    table = pd.DataFrame(rows, index=df.columns)
    table = table.reindex(table["preference beyond ability"].abs()
                          .sort_values(ascending=False).index)
    print("\nstandardized coefficients (how much the feature moves each target)")
    print(f"{'feature':22s} {'real ability':>13} {'preference':>11} {'beyond ability':>15}")
    for f, r in table.iterrows():
        print(f"{f:22s} {r['real ability']:+13.3f} {r['judge preference']:+11.3f} "
              f"{r['preference beyond ability']:+15.3f}")
    table.to_csv(out / "text_features.csv")
    print(f"-> {out / 'text_features.csv'}")


if __name__ == "__main__":
    main()
