"""Verdict-token logits for transcript pairs via vLLM, base or +LoRA adapter.

One jsonl line per pair row with the logprob of "1" and "2" at the
verdict position.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

import prompts


def render_prompt(row, tokenizer):
    msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT_SHORT},
            {"role": "user", "content": prompts.user_text(
                row.question, row.options_1, row.options_2, row.text_1, row.text_2)}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + "ANSWER: "


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--splits", default=None, help="comma list; default all rows")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    base_model = args.base_model or cfg["train"]["base_model"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tid1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    tid2 = tokenizer("2", add_special_tokens=False)["input_ids"][0]

    df = pd.read_parquet(args.pairs)
    if args.splits and "split" in df.columns:
        df = df[df.split.isin(args.splits.split(","))]
    df = df.reset_index(drop=True)

    if args.dry_run:
        print(render_prompt(df.iloc[0], tokenizer))
        print(f"[dry-run] {len(df)} rows")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=base_model,
              max_model_len=cfg["verdicts"]["max_model_len"],
              enable_lora=bool(args.adapter), max_lora_rank=64,
              tensor_parallel_size=args.tp)
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    texts = [render_prompt(r, tokenizer) for r in df.itertuples()]
    lora = LoRARequest("ft", 1, str(args.adapter)) if args.adapter else None
    outs = llm.generate(texts, sp, lora_request=lora)

    n_missing = 0
    with open(args.out, "w") as f:
        for row, out in zip(df.itertuples(), outs):
            lp = out.outputs[0].logprobs[0]
            l1 = lp[tid1].logprob if tid1 in lp else -math.inf
            l2 = lp[tid2].logprob if tid2 in lp else -math.inf
            n_missing += (tid1 not in lp) + (tid2 not in lp)
            rec = dict(pair_uid=row.pair_uid, order=int(row.order), item=row.item,
                       model_1=row.model_1, model_2=row.model_2,
                       label=int(row.label), logit_1=l1, logit_2=l2)
            if "split" in df.columns:
                rec["split"] = row.split
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(df)} rows to {args.out}; verdict token missing from top-20: {n_missing}")


if __name__ == "__main__":
    main()
