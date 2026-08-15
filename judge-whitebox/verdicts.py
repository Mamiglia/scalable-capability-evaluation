"""Verdict runner: batched forward passes reading ANSWER-token logits off the last position."""
import argparse
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wbconfig
import wbpairs
import wbprompts


def readout(logits_last, tokenizer):
    """Logits of "1" and "2" at the last position. logits_last: 1D tensor over vocab."""
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in wbprompts.ANSWER_TOKENS]
    assert all(len(i) == 1 for i in ids), f"answer tokens must be single-token: {ids}"
    (id1,), (id2,) = ids
    return logits_last[id1].item(), logits_last[id2].item()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, adapter, c, dtype=torch.bfloat16):
    model_path = wbconfig.resolve_model(model_name, c)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()

    device = pick_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def run(model_name, adapter, split, limit, out_path, c, invert=False):
    model, tokenizer, device = load_model(model_name, adapter, c)
    df = wbpairs.load_pairs(c, split=split, limit=limit)

    max_tokens = c["verdicts"]["max_input_tokens"]
    batch_size = c["verdicts"]["batch_size"]

    rows, encs = [], []
    n_skipped = 0
    for _, row in df.iterrows():
        prompt = wbprompts.render_verdict_prompt(row, tokenizer, invert=invert)
        enc = tokenizer(prompt)
        if len(enc["input_ids"]) > max_tokens:
            n_skipped += 1
            continue
        rows.append(row)
        encs.append(enc)
    print(f"skipped {n_skipped} rows over {max_tokens} tokens")

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(0, len(encs), batch_size):
            batch_rows = rows[i:i + batch_size]
            batch = tokenizer.pad(encs[i:i + batch_size], return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**batch).logits[:, -1, :]
            for row, row_logits in zip(batch_rows, logits):
                l1, l2 = readout(row_logits, tokenizer)
                pick = 1 if l1 > l2 else 2
                label = int(row["label"])
                target = 3 - label if invert else label
                f.write(json.dumps({
                    "pair_uid": row["pair_uid"],
                    "order": int(row["order"]),
                    "l1": l1,
                    "l2": l2,
                    "pick": pick,
                    "label": label,
                    "correct": pick == target,
                    "invert": invert,
                }) + "\n")


def main():
    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name key into models.hf_ids")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default=c["verdicts"]["eval_split"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--invert", action="store_true",
                        help="ask for the LESS capable model; correct = (pick == 3 - label)")
    args = parser.parse_args()
    run(args.model, args.adapter, args.split, args.limit, args.out, c, invert=args.invert)


if __name__ == "__main__":
    main()
