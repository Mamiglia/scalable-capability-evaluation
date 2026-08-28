"""Fine-tune a small LM to predict a judge's verdict from the pair text (the text-level upper bound).

    python train_verdict_mimic.py --data-dir D --model-dir M --design new-models-and-items-0 --out-dir O

Input per verdict: judge name, benchmark, question, both transcripts as shown (each cut to
--chars-per-side, head and tail kept), classification head over {pick right, pick left}. LoRA on
every linear layer, the head trained fully. Training rows are the design's train rows (both shown
orders are separate rows, so position bias is learnable per judge); the test rows are scored at the
end and written as predictions_judge_verdict_from_pair.csv / held_out_judge_verdict_from_pair.csv
in the GLM fits' layout, plus held_out_by_bench.csv (judge x benchmark).
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
IDENTIFIERS = ["bench", "item", "judge", "model_left", "model_right"]


def cut(text, budget):
    """Keep the head (two thirds) and tail (one third) of a transcript over the character budget."""
    if len(text) <= budget:
        return text
    head, tail = budget * 2 // 3, budget // 3
    return text[:head] + "\n[...]\n" + text[-tail:]


def prompt(row, chars_per_side, question_chars):
    return (f"Judge: {row.judge}\nBenchmark: {row.bench}\n\nQuestion:\n{cut(row.question, question_chars)}\n\n"
            f"Transcript 1:\n{cut(row.text_left, chars_per_side)}\n\nTranscript 2:\n{cut(row.text_right, chars_per_side)}\n\n"
            f"Which transcript is from the more capable model?")


def load_rows(data_dir, design, judges, benches):
    verdicts = pd.read_parquet(data_dir / "verdicts.parquet")
    verdicts["role"] = verdicts[f"role_{design}"]
    verdicts = verdicts[verdicts.role != "drop"]
    if judges:
        verdicts = verdicts[verdicts.judge.isin(judges)]
    if benches:
        verdicts = verdicts[verdicts.bench.isin(benches)]
    transcripts = pd.read_parquet(data_dir / "transcripts.parquet")
    for side in ("left", "right"):
        keys = {k: f"{k}_{side}" if k in ("model", "epoch", "log") else k for k in TRANSCRIPT_KEYS}
        side_table = transcripts.rename(columns={**keys, "transcript": f"text_{side}"})
        if side == "right":
            side_table = side_table.drop(columns="question")
        verdicts = verdicts.merge(side_table, on=list(keys.values()), how="left", validate="many_to_one")
    return verdicts.reset_index(drop=True)


def batches(encoded, labels, order, batch_size, pad_id):
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        width = max(len(encoded[i]) for i in index)
        ids = torch.full((len(index), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(index), width), dtype=torch.long)
        for row, i in enumerate(index):
            ids[row, :len(encoded[i])] = torch.tensor(encoded[i])
            mask[row, :len(encoded[i])] = 1
        yield ids, mask, torch.tensor([labels[i] for i in index], dtype=torch.long), index


def score(model, encoded, labels, batch_size, pad_id, device):
    """P(pick left) per row, rows visited longest first for tight padding."""
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i]))
    out = np.zeros(len(encoded))
    with torch.no_grad():
        for ids, mask, _, index in batches(encoded, labels, order, batch_size, pad_id):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            out[index] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    model.train()
    return out


def report(rows, predicted):
    hit = (predicted > 0.5) == (rows.pick_left.to_numpy() == 1)
    p = np.clip(predicted, 1e-6, 1 - 1e-6)
    y = rows.pick_left.to_numpy()
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    table = [{"group": "all", "n": len(rows), "accuracy": float(hit.mean()), "log_loss": float(loss.mean())}]
    for judge in sorted(rows.judge.unique()):
        m = (rows.judge == judge).to_numpy()
        table.append({"group": judge, "n": int(m.sum()), "accuracy": float(hit[m].mean()), "log_loss": float(loss[m].mean())})
    by_bench = (pd.DataFrame({"judge": rows.judge, "bench": rows.bench, "hit": hit})
                .groupby(["judge", "bench"]).hit.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "accuracy"}))
    return pd.DataFrame(table), by_bench.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judges", nargs="*", default=None)
    parser.add_argument("--benches", nargs="*", default=None)
    parser.add_argument("--chars-per-side", type=int, default=5000)
    parser.add_argument("--question-chars", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--monitor-rows", type=int, default=2000, help="test rows scored every --monitor-every steps")
    parser.add_argument("--monitor-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-rows", type=int, default=None, help="smoke test: first N train and N test rows")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data_dir, args.design, args.judges, args.benches)
    train, test = rows[rows.role == "train"].reset_index(drop=True), rows[rows.role == "test"].reset_index(drop=True)
    if args.limit_rows:
        train, test = train.sample(args.limit_rows, random_state=args.seed), test.sample(args.limit_rows, random_state=args.seed)
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    print(f"{len(train)} train rows, {len(test)} test rows", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    def encode(frame):
        texts = [prompt(r, args.chars_per_side, args.question_chars) for r in frame.itertuples()]
        return tokenizer(texts, truncation=True, max_length=args.max_tokens, add_special_tokens=True)["input_ids"]

    t0 = time.time()
    train_ids, test_ids = encode(train), encode(test)
    print(f"tokenised in {time.time() - t0:.0f}s; train tokens median {np.median([len(x) for x in train_ids]):.0f}, "
          f"max {max(len(x) for x in train_ids)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=2, dtype=torch.bfloat16,
                                                               attn_implementation="sdpa")
    model.config.pad_token_id = pad_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(task_type="SEQ_CLS", r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      modules_to_save=["score"])
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()

    labels = train.pick_left.astype(int).tolist()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    # monitoring rows come out of the training set (the test rows are touched once, at the end)
    monitor = np.random.default_rng(args.seed).choice(len(train_ids), min(args.monitor_rows, len(train_ids) // 10), replace=False)
    monitor_ids, monitor_rows = [train_ids[i] for i in monitor], train.iloc[monitor]
    kept = sorted(set(range(len(train_ids))) - set(monitor.tolist()))
    train_ids, labels = [train_ids[i] for i in kept], [labels[i] for i in kept]
    microbatches_per_epoch = math.ceil(len(train_ids) / args.batch_size)
    total_micro = int(microbatches_per_epoch * args.epochs)
    steps_per_epoch = math.ceil(len(train_ids) / (args.batch_size * args.grad_accumulation))
    total_steps = int(steps_per_epoch * args.epochs)
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.03 * total_steps), total_steps)

    step, micro, running, group = 0, 0, [], 0
    model.train()
    t0 = time.time()
    while micro < total_micro:
        # length-bucketed shuffling: shuffle, then sort within chunks of 50 batches so padding stays small
        order = list(range(len(train_ids)))
        random.shuffle(order)
        chunk = args.batch_size * 50
        order = [i for start in range(0, len(order), chunk)
                 for i in sorted(order[start:start + chunk], key=lambda i: len(train_ids[i]))]
        for ids, mask, y, _ in batches(train_ids, labels, order, args.batch_size, pad_id):
            micro += 1
            group_size = min(args.grad_accumulation, total_micro - (micro - 1) // args.grad_accumulation * args.grad_accumulation)
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            (out.loss / group_size).backward()          # a final partial group is averaged over its own size
            running.append(out.loss.item())
            group += 1
            if group == group_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                schedule.step()
                optimiser.zero_grad(set_to_none=True)
                step, group = step + 1, 0
                if step % 50 == 0:
                    print(f"step {step}/{total_steps} loss {np.mean(running[-50 * args.grad_accumulation:]):.4f} "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)
                if step % args.monitor_every == 0 or micro >= total_micro:
                    predicted = score(model, monitor_ids, monitor_rows.pick_left.tolist(), args.eval_batch_size, pad_id, device)
                    print(f"  monitor accuracy {((predicted > 0.5) == (monitor_rows.pick_left.to_numpy() == 1)).mean():.3f} "
                          f"on {len(monitor_ids)} held-back training rows", flush=True)
            if micro >= total_micro:
                break

    predicted = score(model, test_ids, test.pick_left.tolist(), args.eval_batch_size, pad_id, device)
    scores, by_bench = report(test, predicted)
    test[IDENTIFIERS].assign(observed=test.pick_left.to_numpy(), predicted=predicted).to_csv(
        args.out_dir / "predictions_judge_verdict_from_pair.csv", index=False)
    scores.to_csv(args.out_dir / "held_out_judge_verdict_from_pair.csv", index=False)
    by_bench.to_csv(args.out_dir / "held_out_by_bench.csv", index=False)
    model.save_pretrained(args.out_dir / "adapter")
    print(scores.to_string(index=False))
    print(by_bench.to_string(index=False))


if __name__ == "__main__":
    main()
