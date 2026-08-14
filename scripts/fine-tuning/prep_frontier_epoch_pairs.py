"""Frontier saturation pairs from the epoch-pool GPQA logs (full reasoning
budgets), instead of fresh low-effort generation. Held-out items only:
train-item transcripts of these authors appear in the judge's training set.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from prep_pairs import check_counterbalance, make_pair_rows, token_filter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=["claude-opus-4-6_32K", "gemini-3.1-pro-preview",
                             "gpt-5.5-pre-release_xhigh"])
    ap.add_argument("--models-b", nargs="+", default=None,
                    help="second tier: build only cross-tier pairs (models x models-b)")
    ap.add_argument("--split-name", default="frontier_gpqa_epoch")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--out", default="judge_results/judge_ft/pairs_frontier_epoch.parquet")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["out_dir"])

    accs = pd.read_csv(out_dir / "models_ft.csv").set_index("model").accuracy
    split = pd.read_csv(out_dir / "item_split.csv")
    holdout = set(split.item[split.holdout])

    all_models = args.models + (args.models_b or [])
    frames = {m: pd.read_parquet(out_dir / "transcripts_cache" / f"{m}.parquet")
              for m in all_models}
    rows = []
    for item in sorted(holdout):
        metas = {}
        for m, f in frames.items():
            hit = f[f.item == item]
            if len(hit):
                r = hit.iloc[0]
                metas[m] = dict(model=m, cap=float(accs[m]),
                                text=r.text[:cfg["prep"]["max_len_chars"]],
                                options=r.options, question=r.question)
        if args.models_b:
            combos = [(a, b) for a in args.models for b in args.models_b
                      if a in metas and b in metas]
        else:
            present = sorted(m for m in args.models if m in metas)
            combos = [(present[i], present[j]) for i in range(len(present))
                      for j in range(i + 1, len(present))]
        for a, b in combos:
            pair = make_pair_rows(item, metas[a], metas[b], question=metas[a]["question"])
            for row in pair:
                row["split"] = args.split_name
            rows.extend(pair)
    pairs = pd.DataFrame(rows)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["prep"]["tokenizer"])
    pairs, n_over = token_filter(pairs, tok, cfg["prep"]["max_input_tokens"])
    check_counterbalance(pairs)
    pairs.to_parquet(args.out)
    print(f"pairs: {pairs.pair_uid.nunique()} on {pairs.item.nunique()} held-out items "
          f"({n_over} dropped for length)")
    print("caps:", {m: round(float(accs[m]), 3) for m in args.models})


if __name__ == "__main__":
    main()
