"""Phase-2 (rewrite-aug LOO) verdicts. For each held-out domain d, score on d's
plain eval_mm and d's adversarial style-swap eval:
  base (floor), plain-LOO judge (loo_d), aug-LOO judge (loo_d_aug).
One model load. New jsonls only. Run on SWC. Label = indomain."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pod_verdicts import render_prompt

DOMAINS = ["gpqa", "aime", "chess"]
LABEL = "indomain"


def _score(llm, sp, texts, tid1, tid2, lora):
    outs = llm.generate(texts, sp, lora_request=lora)
    r = []
    for out in outs:
        lp = out.outputs[0].logprobs[0]
        r.append((lp[tid1].logprob if tid1 in lp else -math.inf,
                  lp[tid2].logprob if tid2 in lp else -math.inf))
    return r


def _write(path, df, scored):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row, (l1, l2) in zip(df.itertuples(), scored):
            f.write(json.dumps(dict(pair_uid=row.pair_uid, order=int(row.order),
                    item=str(row.item), model_1=row.model_1, model_2=row.model_2,
                    label=int(row.label), logit_1=l1, logit_2=l2,
                    split=row.split, domain=row.domain)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-root", required=True)      # cross/<size>
    ap.add_argument("--pairs-dir", required=True)     # cross_domain
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--vcfg", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    out_root, pd_dir = Path(args.out_root), Path(args.pairs_dir)
    vcfg = __import__("yaml").safe_load(open(args.vcfg))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tid1 = tok("1", add_special_tokens=False)["input_ids"][0]
    tid2 = tok("2", add_special_tokens=False)["input_ids"][0]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.base_model, max_model_len=vcfg["verdicts"]["max_model_len"],
              enable_lora=True, max_lora_rank=64, tensor_parallel_size=args.tp)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    ph = out_root / "phase2"
    lid = 1

    for d in DOMAINS:
        style = pd.read_parquet(pd_dir / f"pairs_style_{d}_{LABEL}.parquet").reset_index(drop=True)
        stexts = [render_prompt(r, tok) for r in style.itertuples()]
        plain = pd.read_parquet(pd_dir / f"pairs_{d}_{LABEL}.parquet")
        plain = plain[plain.split == "eval_mm"].reset_index(drop=True)
        ptexts = [render_prompt(r, tok) for r in plain.itertuples()]

        plain_ad = out_root / f"loo_{d}_{LABEL}" / "adapters" / "final"
        aug_ad = out_root / f"loo_{d}_{LABEL}_aug" / "adapters" / "final"
        jobs = [("base", None, style, stexts, ph / f"base_style_{d}.jsonl"),
                ("plainloo", plain_ad, style, stexts, ph / f"plainloo_style_{d}.jsonl"),
                ("augloo", aug_ad, style, stexts, ph / f"augloo_style_{d}.jsonl"),
                ("augloo", aug_ad, plain, ptexts, ph / f"augloo_plain_{d}.jsonl")]
        for name, ad, df, txts, out in jobs:
            if out.exists():
                print(f"have {out.name}"); continue
            lora = None
            if ad is not None:
                if not (ad / "adapter_config.json").exists():
                    print(f"skip {out.name}: no adapter"); continue
                lora = LoRARequest(str(ad), lid, str(ad)); lid += 1
            _write(out, df, _score(llm, sp, txts, tid1, tid2, lora))
            print(f"scored {out.name} ({len(df)} rows)")
    print("PHASE2_VERDICTS_DONE")


if __name__ == "__main__":
    main()
