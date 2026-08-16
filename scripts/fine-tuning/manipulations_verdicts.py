"""Score manipulation pair tables with a base or rank-1 fine-tuned judge.

Uses the exact short training prompt and the logits of verdict tokens 1 and 2.
One invocation handles one of four condition shards for one model variant.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

import prompts

CONDITIONS = (
    "trunc_first_10", "trunc_first_50", "trunc_first_90", "trunc_first_100",
    "trunc_end_10", "trunc_end_50", "trunc_end_90", "trunc_end_100",
    "paraphrase_haiku", "paraphrase_gpt",
)


def conditions_for_shard(shard):
    if shard not in range(4):
        raise ValueError("shard must be 0, 1, 2, or 3")
    return CONDITIONS[shard::4]


def system_prompt(condition):
    if condition.startswith("trunc_first_"):
        return prompts.SYSTEM_PROMPT_SHORT + prompts.TRUNCATION_NOTE
    if condition.startswith("trunc_end_"):
        return prompts.SYSTEM_PROMPT_SHORT + prompts.TRUNCATION_NOTE_END
    return prompts.SYSTEM_PROMPT_SHORT


def render_prompt(row, tokenizer, condition):
    messages = [
        {"role": "system", "content": system_prompt(condition)},
        {"role": "user", "content": prompts.user_text(
            row.question, row.options_1, row.options_2, row.text_1, row.text_2)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + "ANSWER: "


def filter_overlength(frame, texts, tokenizer, maximum, batch_size=128):
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], add_special_tokens=False)
        lengths.extend(map(len, encoded["input_ids"]))
    over = set(frame.loc[[n > maximum for n in lengths], "pair_uid"])
    keep = ~frame.pair_uid.isin(over)
    return frame.loc[keep].reset_index(drop=True), [t for t, ok in zip(texts, keep) if ok], len(over)


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_records(path, records):
    """Resume one deterministic partial, validate its prefix, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if path.exists():
        if read_records(path) != records:
            raise ValueError(f"existing completed output differs: {path}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    existing = read_records(partial) if partial.exists() else []
    if existing != records[:len(existing)]:
        raise ValueError(f"partial output is not a prefix of this run: {partial}")
    with partial.open("a") as handle:
        for record in records[len(existing):]:
            handle.write(json.dumps(record) + "\n")
    if read_records(partial) != records:
        raise ValueError(f"failed to validate completed partial: {partial}")
    partial.replace(path)


def valid_output(path, frame):
    if not path.exists():
        return False
    try:
        records = read_records(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = set(zip(frame.pair_uid.astype(str), frame.order.astype(int)))
    observed = {(str(row["pair_uid"]), int(row["order"])) for row in records}
    return len(records) == len(frame) and observed == expected


def scored_records(frame, outputs, token_1, token_2, condition, variant):
    records, missing = [], 0
    for row, output in zip(frame.itertuples(), outputs):
        logprobs = output.outputs[0].logprobs[0]
        logit_1 = logprobs[token_1].logprob if token_1 in logprobs else -math.inf
        logit_2 = logprobs[token_2].logprob if token_2 in logprobs else -math.inf
        missing += (token_1 not in logprobs) + (token_2 not in logprobs)
        pick = 1 if logit_1 > logit_2 else 2
        records.append({
            "condition": condition, "variant": variant,
            "pair_uid": str(row.pair_uid), "order": int(row.order),
            "item": str(row.item), "model_1": row.model_1, "model_2": row.model_2,
            "label": int(row.label), "logit_1": logit_1, "logit_2": logit_2,
            "pick": pick, "correct": pick == int(row.label),
        })
    return records, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=28672)
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    token_2 = tokenizer("2", add_special_tokens=False)["input_ids"][0]
    pending = []
    for condition in conditions_for_shard(args.shard):
        frame = pd.read_parquet(args.pairs_dir / f"{condition}.parquet")
        texts = [render_prompt(row, tokenizer, condition) for row in frame.itertuples()]
        frame, texts, dropped = filter_overlength(
            frame, texts, tokenizer, args.max_input_tokens)
        output = args.out_root / args.variant / f"{condition}.jsonl"
        print(f"{condition}: {len(frame)} rows; {dropped} overlength pair-items dropped", flush=True)
        if valid_output(output, frame):
            print(f"{condition}: complete, skipping", flush=True)
        else:
            pending.append((condition, frame, texts, output))
    if not pending:
        print("MANIPULATION_VERDICTS_DONE", flush=True)
        return

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.base_model, max_model_len=args.max_model_len,
        enable_lora=bool(args.adapter), max_lora_rank=64,
        tensor_parallel_size=1, gpu_memory_utilization=0.92,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    lora = LoRARequest("ft", 1, args.adapter) if args.adapter else None
    for condition, frame, texts, output in pending:
        generated = llm.generate(texts, sampling, lora_request=lora)
        records, missing = scored_records(
            frame, generated, token_1, token_2, condition, args.variant)
        write_records(output, records)
        print(f"{condition}: wrote {len(records)} rows; {missing} missing verdict logits", flush=True)
    print("MANIPULATION_VERDICTS_DONE", flush=True)


if __name__ == "__main__":
    main()
