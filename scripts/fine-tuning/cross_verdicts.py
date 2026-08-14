"""Cross-domain verdict sweep. Loads the base model ONCE (vLLM + LoRA) and scores
every trained cell adapter — plus the base-model floor — on each test domain's
held-out eval set. One jsonl per (cell, label, test-domain). Run on SWC.

  python cross_verdicts.py --base-model <path> --out-root <ceph>/cross/<size> \
      --pairs-dir <ceph>/cross_domain --tp 1
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from pod_verdicts import render_prompt   # reuse the exact prompt rendering


def _score(llm, sp, texts, tid1, tid2, lora):
    outs = llm.generate(texts, sp, lora_request=lora)
    rows = []
    for out in outs:
        lp = out.outputs[0].logprobs[0]
        l1 = lp[tid1].logprob if tid1 in lp else -math.inf
        l2 = lp[tid2].logprob if tid2 in lp else -math.inf
        rows.append((l1, l2))
    return rows


def _write(path, df, scored):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row, (l1, l2) in zip(df.itertuples(), scored):
            rec = dict(pair_uid=row.pair_uid, order=int(row.order), item=str(row.item),
                       model_1=row.model_1, model_2=row.model_2, label=int(row.label),
                       logit_1=l1, logit_2=l2, split=row.split, domain=row.domain)
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-root", required=True, help="ceph cross/<size> dir")
    ap.add_argument("--pairs-dir", required=True, help="ceph cross_domain dir with pairs_*.parquet")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--eval-split", default="eval_mm", help="held-out split to score")
    ap.add_argument("--config", default=str(Path(__file__).parent / "cross_domain.yaml"))
    ap.add_argument("--vcfg", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    vcfg = yaml.safe_load(open(args.vcfg))
    out_root, pairs_dir = Path(args.out_root), Path(args.pairs_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tid1 = tok("1", add_special_tokens=False)["input_ids"][0]
    tid2 = tok("2", add_special_tokens=False)["input_ids"][0]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.base_model, max_model_len=vcfg["verdicts"]["max_model_len"],
              enable_lora=True, max_lora_rank=64, tensor_parallel_size=args.tp)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    # render each test set's prompts once
    evalset = {}
    for label in cfg["labels"]:
        for dom in cfg["domains"]:
            df = pd.read_parquet(pairs_dir / f"pairs_{dom}_{label}.parquet")
            df = df[df.split == args.eval_split].reset_index(drop=True)
            evalset[(dom, label)] = (df, [render_prompt(r, tok) for r in df.itertuples()])

    # stable LoRA id per adapter (reused across the domains it is scored on)
    names = [f"{c}_{l}" for l in cfg["labels"] for c in cfg["cells"]]
    lora_id = {n: i + 1 for i, n in enumerate(names)}

    # base-model floor, once per test set
    for (dom, label), (df, texts) in evalset.items():
        out = out_root / "base" / f"eval_{dom}_{label}.jsonl"
        if not out.exists():
            _write(out, df, _score(llm, sp, texts, tid1, tid2, None))
            print(f"base floor: {dom}/{label} ({len(df)} rows)")

    # each cell adapter on each test domain (of its label)
    for label in cfg["labels"]:
        for cell in cfg["cells"]:
            name = f"{cell}_{label}"
            adapter = out_root / name / "adapters" / "final"
            if not (adapter / "adapter_config.json").exists():
                print(f"skip {name}: adapter missing")
                continue
            lora = LoRARequest(name, lora_id[name], str(adapter))
            for dom in cfg["domains"]:
                out = out_root / name / f"eval_{dom}.jsonl"
                if out.exists():
                    continue
                df, texts = evalset[(dom, label)]
                _write(out, df, _score(llm, sp, texts, tid1, tid2, lora))
                print(f"scored {name} on {dom} ({len(df)} rows)")
    print("CROSS_VERDICTS_DONE")


if __name__ == "__main__":
    main()
