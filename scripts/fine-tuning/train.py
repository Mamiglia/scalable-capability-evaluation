"""LoRA-SFT a judge to pick the more capable author of a transcript pair.

Cross-entropy on the verdict token only. Standalone for pod use: imports
only the sibling prompts module.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

import prompts


def render_example(row, tokenizer, max_input_tokens):
    """Token ids for prompt + verdict token; loss position is the last id."""
    msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT_SHORT},
            {"role": "user", "content": prompts.user_text(
                row.question, row.options_1, row.options_2, row.text_1, row.text_2)}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text + "ANSWER: ", add_special_tokens=False)["input_ids"]
    label_ids = tokenizer(str(row.label), add_special_tokens=False)["input_ids"]
    assert len(label_ids) == 1
    ids = ids[:max_input_tokens] + label_ids
    return ids, len(ids) - 1


def loss_plateaued(recent_losses, step, check_steps, threshold):
    """True when past the grace period and the recent mean sits at or above
    chance-level CE — the collapsed-cell signature (loss pinned near ln 2)."""
    if step < check_steps or not recent_losses:
        return False
    return sum(recent_losses) / len(recent_losses) > threshold


def verdict_token_ids(tokenizer):
    ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in ("1", "2")]
    assert all(len(x) == 1 for x in ids)
    return ids[0][0], ids[1][0]


@torch.no_grad()
def val_accuracy(model, rows, tokenizer, max_input_tokens, tid1, tid2):
    model.eval()
    correct = 0
    for row in rows.itertuples():
        ids, _ = render_example(row, tokenizer, max_input_tokens)
        input_ids = torch.tensor([ids[:-1]], device=model.device)
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[0, -1]
        pred = 1 if logits[tid1] > logits[tid2] else 2
        correct += pred == row.label
    model.train()
    return correct / len(rows)


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--resume-adapter", default=None,
                    help="continue training an existing LoRA adapter (rank/alpha come from it)")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="train on first N rows (smoke)")
    ap.add_argument("--splits", default="train", help="comma list of splits to train on")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default=None, help="override config train.dtype (CPU smoke: float32)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["train"]
    base = args.base_model or cfg["base_model"]
    max_input_tokens = yaml.safe_load(open(args.config))["prep"]["max_input_tokens"]
    grad_accum = cfg["grad_accum"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    df = pd.read_parquet(args.pairs)
    train_splits = args.splits.split(",")
    rows = df[df.split.isin(train_splits)].sample(frac=1, random_state=cfg["seed"])
    if args.limit:
        rows = rows.iloc[:args.limit]
    val_rows = df[df.split == "val"] if "val" not in train_splits else df.iloc[:0]
    if args.limit:
        val_rows = val_rows.iloc[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(base)
    tid1, tid2 = verdict_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=getattr(torch, args.dtype or cfg["dtype"]), device_map=args.device_map)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        assert args.lora_rank, "--lora-rank required unless --resume-adapter"
        peft_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=cfg["lora_alpha_factor"] * args.lora_rank,
            lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
            task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    from collections import deque
    recent = deque(maxlen=cfg["abort_window"])
    aborted = False
    n_skipped, tokens_seen, loss_sum, pending = 0, 0, 0.0, False
    t0 = time.time()
    for step, row in enumerate(rows.itertuples()):
        ids, pos = render_example(row, tokenizer, max_input_tokens)
        if len(ids) > max_input_tokens + 1:
            n_skipped += 1
            continue
        input_ids = torch.tensor([ids[:-1]], device=model.device)
        # logits_to_keep=1: full-sequence vocab logits would dominate peak memory
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        loss = F.cross_entropy(out.logits[0, -1:].float(),
                               torch.tensor([ids[-1]], device=model.device))
        (loss / grad_accum).backward()
        pending = True
        if (step + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad()
            pending = False
        tokens_seen += len(ids)
        loss_sum += loss.item()
        recent.append(loss.item())
        if loss_plateaued(recent, step, cfg["abort_check_steps"], cfg["abort_loss_threshold"]):
            print(f"ABORT at step {step + 1}: recent mean loss "
                  f"{sum(recent) / len(recent):.3f} still at chance level", flush=True)
            aborted = True
            break
        if (step + 1) % 20 == 0:
            rec = dict(step=step + 1, loss=round(loss_sum / 20, 4), lr=args.lr,
                       tokens=tokens_seen, sec=round(time.time() - t0, 1))
            print(rec, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            loss_sum = 0.0
    if pending:
        opt.step()
        opt.zero_grad()

    final = dict(n_train=len(rows) - n_skipped, n_skipped=n_skipped, tokens=tokens_seen,
                 sec=round(time.time() - t0, 1), aborted=aborted)
    if len(val_rows):
        final["val_acc"] = round(val_accuracy(model, val_rows, tokenizer,
                                              max_input_tokens, tid1, tid2), 4)
    model.save_pretrained(out_dir)
    print(final, flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(final) + "\n")


if __name__ == "__main__":
    main()
