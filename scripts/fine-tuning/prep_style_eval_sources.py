"""Adversarial style-robustness eval on HELD-OUT pairs (eval_mm = held-out models
x held-out items). Restyle each pair so the writing style fights capability: the
WEAKER author gets a strong model's voice, the STRONGER author a weak model's
plain voice. A style-robust judge still picks the stronger author; a style-reliant
one flips. The style analog of the sandbag.

Writes style_eval_sources.parquet (for pod_rewrite) and style_eval_meta.parquet
(pair metadata to rebuild the eval pairs after rewriting).
"""

import argparse

import pandas as pd


def strip_ans(t):
    i = t.rfind("ANSWER:")
    return t[:i].rstrip() if i != -1 else t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="judge_results/judge_ft/pairs.parquet")
    ap.add_argument("--models", default="judge_results/judge_ft/models_ft.csv")
    ap.add_argument("--out-sources", default="judge_results/judge_ft/style_eval_sources.parquet")
    ap.add_argument("--out-meta", default="judge_results/judge_ft/style_eval_meta.parquet")
    ap.add_argument("--n-pairs", type=int, default=250)
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = pd.read_parquet(args.pairs)
    ev = p[p.split == "eval_mm"].drop_duplicates("pair_uid")   # one order-0 row per pair
    ev = ev.sample(min(args.n_pairs, len(ev)), random_state=args.seed)

    # strong / weak voice-source models (by capability) with >=4 train transcripts
    tr = p[p.split == "train"]
    by_model = {}
    for r in tr.itertuples():
        by_model.setdefault(r.model_1, []).append(r.text_1)
    logit = pd.read_csv(args.models).set_index("model").logit
    usable = [m for m in by_model if len(by_model[m]) >= 4 and m in logit.index]
    usable.sort(key=lambda m: logit[m])
    weak_voice, strong_voice = usable[0], usable[-1]

    def exemplars(m):
        return [t for t in sorted(by_model[m], key=len) if 300 <= len(t) <= 1800][:args.n_examples] \
               or by_model[m][:args.n_examples]
    strong_ex, weak_ex = exemplars(strong_voice), exemplars(weak_voice)

    srcs, meta = [], []
    for r in ev.itertuples():
        strong_is_1 = r.cap_1 >= r.cap_2
        s_text, w_text = (r.text_1, r.text_2) if strong_is_1 else (r.text_2, r.text_1)
        # stronger -> weak voice (plain); weaker -> strong voice (impressive)
        srcs.append(dict(id=f"{r.pair_uid}|S", model=r.model_1, item=str(r.item),
                         text=s_text, target_voice=weak_voice, examples=weak_ex))
        srcs.append(dict(id=f"{r.pair_uid}|W", model=r.model_2, item=str(r.item),
                         text=w_text, target_voice=strong_voice, examples=strong_ex))
        meta.append(dict(pair_uid=r.pair_uid, item=str(r.item), model_1=r.model_1,
                         model_2=r.model_2, cap_1=r.cap_1, cap_2=r.cap_2, label=int(r.label),
                         question=r.question, options_1=r.options_1, options_2=r.options_2,
                         text_1=r.text_1, text_2=r.text_2, strong_is_1=bool(strong_is_1)))
    pd.DataFrame(srcs).to_parquet(args.out_sources)
    pd.DataFrame(meta).to_parquet(args.out_meta)
    print(f"{len(meta)} eval pairs -> {len(srcs)} rewrites | strong-voice={strong_voice} "
          f"(logit {logit[strong_voice]:.2f}) weak-voice={weak_voice} (logit {logit[weak_voice]:.2f})")


if __name__ == "__main__":
    main()
