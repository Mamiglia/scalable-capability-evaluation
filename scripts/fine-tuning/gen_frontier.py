"""Generate frontier-model answers on GPQA + LAB-Bench via OpenRouter.

Resumable (skips rows already in the output jsonl), records the serving
provider per response, aborts hard when projected cost exceeds the budget.
LAB-Bench image subtasks attach their figures/tables as base64 parts.
"""

import argparse
import asyncio
import base64
import io
import json
import random
import re
import string
from pathlib import Path

import pandas as pd
import yaml

LETTERS = string.ascii_uppercase
ANSWER_RE = re.compile(r"ANSWER:\s*([A-Z])")

# LAB-Bench official multiple-choice prompt, verbatim from the paper harness
LAB_TEMPLATE = (
    "The following is a multiple choice question about biology.\n"
    "Please answer by responding with the letter of the correct answer.\n"
    "\n"
    "Think step by step.\n"
    "\n"
    "Question: {question}\n"
    "Options:\n"
    "{choices}\n"
    "\n"
    "You MUST include the letter of the correct answer within the following "
    "format: 'ANSWER: $LETTER' (without quotes). For example, 'ANSWER: "
    "<answer>', where <answer> is the correct letter. Always answer in "
    "exactly this format of a single letter, even if you are unsure. We "
    "require this because we use automatic parsing."
)

GPQA_SYSTEM = "Answer the multiple-choice question. End your reply with 'ANSWER: <letter>'."


def b64_part(img) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def gpqa_tasks(cfg, seed):
    df = pd.read_parquet(cfg["gpqa"]["items_parquet"])
    df = df.sample(n=cfg["gpqa"]["n_items"], random_state=seed).reset_index(drop=True)
    tasks = []
    for r in df.itertuples():
        text = f"{r.question}\n\n{r.options_rendered}"
        for epoch in range(1, cfg["epochs"] + 1):
            tasks.append(dict(eval="gpqa", item=r.item, subtask="gpqa_diamond",
                              epoch=epoch, target=r.target_letter,
                              system=GPQA_SYSTEM,
                              content=[{"type": "text", "text": text}]))
    return tasks


def labbench_tasks(cfg, seed):
    from datasets import load_dataset
    lb = cfg["labbench"]
    tasks = []
    for kind, subtasks, budget in [("text", lb["text_subtasks"], lb["n_text"]),
                                   ("image", lb["image_subtasks"], lb["n_image"])]:
        records = []
        for sub in subtasks:
            for r in load_dataset(lb["source"], sub, split="train"):
                qlen = len(r["question"]) + sum(len(d) for d in r["distractors"])
                if qlen > lb["max_question_chars"]:
                    continue
                records.append((sub, r))
        records.sort(key=lambda t: str(t[1]["id"]))
        random.Random(seed).shuffle(records)
        records = records[:budget]
        for sub, r in records:
            for epoch in range(1, cfg["epochs"] + 1):
                options = [r["ideal"], *r["distractors"]]
                random.Random((seed, str(r["id"]), epoch).__hash__()).shuffle(options)
                choices = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
                content = [{"type": "text", "text": LAB_TEMPLATE.format(
                    question=r["question"], choices=choices)}]
                imgs = ([r["figure"]] if sub == "FigQA"
                        else r["tables"] if sub == "TableQA" else [])
                content += [b64_part(im) for im in imgs]
                tasks.append(dict(eval="labbench", item=str(r["id"]), subtask=sub,
                                  epoch=epoch, target=LETTERS[options.index(r["ideal"])],
                                  system=None, content=content))
    return tasks


async def run(cfg, env_file):
    import os
    from openai import AsyncOpenAI
    key = os.environ.get("OPENROUTER_API_KEY") or next(
        l.split("=", 1)[1].strip() for l in open(env_file)
        if l.startswith("OPENROUTER_API_KEY="))
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    out = Path(cfg["out_jsonl"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["eval"], r["item"], r["epoch"], r["model"]))

    tasks = gpqa_tasks(cfg, cfg["seed"]) + labbench_tasks(cfg, cfg["seed"])
    todo = [(m, t) for t in tasks for m in cfg["models"]
            if (t["eval"], t["item"], t["epoch"], m) not in done]
    random.Random(cfg["seed"]).shuffle(todo)
    print(f"{len(tasks)} (item, epoch) tasks x {len(cfg['models'])} models; "
          f"{len(done)} done, {len(todo)} to go")

    spent = 0.0
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(cfg["concurrency"])

    async def one(model, t):
        nonlocal spent
        mc = cfg["models"][model]
        msgs = ([{"role": "system", "content": t["system"]}] if t["system"] else []) + \
               [{"role": "user", "content": t["content"]}]
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=mc["slug"], messages=msgs,
                        temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                        extra_body={"provider": {"data_collection": "deny"},
                                    "reasoning": {"effort": mc["reasoning_effort"]},
                                    "usage": {"include": True}},
                        timeout=300)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                price = cfg["pricing_per_m"][model]
                cost = (usage.prompt_tokens * price["input"]
                        + usage.completion_tokens * price["output"]) / 1e6
                m = ANSWER_RE.findall(text)
                letter = m[-1] if m else None
                row = dict(eval=t["eval"], item=t["item"], subtask=t["subtask"],
                           epoch=t["epoch"], model=model, target=t["target"],
                           letter=letter, correct=letter == t["target"],
                           finished=bool(letter), text=text,
                           provider=getattr(resp, "provider", None),
                           gen_id=resp.id, prompt_tokens=usage.prompt_tokens,
                           completion_tokens=usage.completion_tokens, cost=round(cost, 6))
                async with lock:
                    spent += cost
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if spent > cfg["budget_usd"]:
                        raise SystemExit(f"BUDGET EXCEEDED: ${spent:.2f}")
                return
            except SystemExit:
                raise
            except Exception as e:
                if attempt == 3:
                    print(f"FAILED {model} {t['eval']}/{t['item']}/e{t['epoch']}: {e}")
                    return
                await asyncio.sleep(5 * (attempt + 1))

    for i in range(0, len(todo), 200):
        await asyncio.gather(*(one(m, t) for m, t in todo[i:i + 200]))
        print(f"progress {min(i + 200, len(todo))}/{len(todo)}  spent ${spent:.2f}", flush=True)
    print(f"DONE  total new spend ${spent:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "frontier_config.yaml"))
    ap.add_argument("--env-file", default=".env", help="file holding OPENROUTER_API_KEY")
    args = ap.parse_args()
    asyncio.run(run(yaml.safe_load(open(args.config)), args.env_file))
