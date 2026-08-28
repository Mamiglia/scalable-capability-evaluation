"""Which parts of the two transcripts move the mimic's verdict: leave-one-unit-out attribution.

    python attribute_units.py --data-dir D --model-dir M --adapter A --design model-fold-2 --benches gpqa --out F.parquet [--pairs 400]

For each held-out verdict row, the transcripts are split into units (paragraphs, or sentences when a
paragraph is long); the mimic's logit for "pick left" is recomputed with each unit removed, and the
drop is that unit's attribution (positive = the unit pushes the judge towards the side it sits on).
Writes one row per unit: verdict keys, side, unit index, text, base logit, logit without the unit.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_verdict_mimic import cut, load_rows

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\(\[])")
LONG_PARAGRAPH = 600


def units_of(text, budget):
    text = cut(text, budget)
    out = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > LONG_PARAGRAPH:
            out += [s for s in SENTENCE.split(paragraph) if s.strip()]
        else:
            out.append(paragraph)
    return out


def prompt_from_units(row, left_units, right_units, question_chars):
    return (f"Judge: {row.judge}\nBenchmark: {row.bench}\n\nQuestion:\n{cut(row.question, question_chars)}\n\n"
            f"Transcript 1:\n{chr(10).join(left_units)}\n\nTranscript 2:\n{chr(10).join(right_units)}\n\n"
            f"Which transcript is from the more capable model?")


@torch.no_grad()
def logits(model, tokenizer, texts, max_tokens, batch_size, device):
    out = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_tokens, padding=True, return_tensors="pt")
        z = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)).logits.float()
        out += (z[:, 1] - z[:, 0]).cpu().tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--benches", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--chars-per-side", type=int, default=5000)
    parser.add_argument("--question-chars", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda"
    rows = load_rows(args.data_dir, args.design, None, args.benches)
    test = rows[rows.role == "test"].sample(min(args.pairs, int((rows.role == "test").sum())), random_state=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    records = []
    for i, row in enumerate(test.itertuples()):
        left, right = units_of(row.text_left, args.chars_per_side), units_of(row.text_right, args.chars_per_side)
        texts = [prompt_from_units(row, left, right, args.question_chars)]
        for k in range(len(left)):
            texts.append(prompt_from_units(row, left[:k] + left[k + 1:], right, args.question_chars))
        for k in range(len(right)):
            texts.append(prompt_from_units(row, left, right[:k] + right[k + 1:], args.question_chars))
        z = logits(model, tokenizer, texts, args.max_tokens, args.batch_size, device)
        keys = {"bench": row.bench, "item": row.item, "judge": row.judge, "model_left": row.model_left, "model_right": row.model_right,
                "epoch_left": row.epoch_left, "epoch_right": row.epoch_right, "pick_left": row.pick_left, "base_logit": z[0]}
        for k, unit in enumerate(left):
            records.append({**keys, "side": "left", "unit": k, "n_units": len(left), "text": unit, "logit_without": z[1 + k]})
        for k, unit in enumerate(right):
            records.append({**keys, "side": "right", "unit": k, "n_units": len(right), "text": unit, "logit_without": z[1 + len(left) + k]})
        if i % 50 == 0:
            print(f"{i}/{len(test)} pairs", flush=True)
    table = pd.DataFrame(records)
    # attribution towards the unit's own side: removing a left unit lowers the left logit if it helped left
    table["attribution"] = np.where(table.side == "left", table.base_logit - table.logit_without, table.logit_without - table.base_logit)
    table.to_parquet(args.out, index=False)
    print(f"{len(table)} units from {len(test)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
