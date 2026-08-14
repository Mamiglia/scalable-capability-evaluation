"""GPQA answering accuracy of base or fine-tuned model via vLLM sampling.

One jsonl line per (item, epoch) with the parsed answer letter.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ANSWER_SYSTEM = "Answer the multiple-choice question. End your reply with 'ANSWER: <letter>'."
ANSWER_RE = re.compile(r"ANSWER:\s*([A-D])")


def render_prompt(row, tokenizer):
    msgs = [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{row.question}\n\n{row.options_rendered}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    g = cfg["gpqa"]
    base_model = args.base_model or cfg["train"]["base_model"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    df = pd.read_parquet(args.items).reset_index(drop=True)

    if args.dry_run:
        print(render_prompt(df.iloc[0], tokenizer))
        print(f"[dry-run] {len(df)} items x {g['epochs']} epochs")
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=base_model,
              max_model_len=cfg["verdicts"]["max_model_len"],
              enable_lora=bool(args.adapter), max_lora_rank=64,
              tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=g["temperature"], top_p=g["top_p"],
                        max_tokens=g["max_new_tokens"], n=g["epochs"], seed=g["seed"])
    texts = [render_prompt(r, tokenizer) for r in df.itertuples()]
    lora = LoRARequest("ft", 1, str(args.adapter)) if args.adapter else None
    outs = llm.generate(texts, sp, lora_request=lora)

    with open(args.out, "w") as f:
        for row, out in zip(df.itertuples(), outs):
            for epoch, comp in enumerate(out.outputs):
                matches = ANSWER_RE.findall(comp.text)
                letter = matches[-1] if matches else None
                f.write(json.dumps(dict(
                    item=row.item, epoch=epoch, letter=letter,
                    target=row.target_letter,
                    correct=bool(letter and letter == row.target_letter),
                    finished=bool(letter))) + "\n")
    print(f"wrote {len(df) * g['epochs']} rows to {args.out}")


if __name__ == "__main__":
    main()
