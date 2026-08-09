"""Run style_capability generation on a remote CPU host via direct OpenRouter
calls (no inspect, no GPU). Reads payload.json (runs + params) and
items.parquet, writes answers.jsonl. Resumable: existing non-error
(target,label,sample_id,epoch) rows are skipped. One thread pool over all
(run,item,epoch) tasks -> requests spread across all pinned providers at once.

Env: OPENROUTER_API_KEY. Deps: openai, pandas, pyarrow.
  python pod_gen.py --payload payload.json --items items.parquet \
      --out answers.jsonl --concurrency 60
"""

import argparse
import json
import os
import random
import re
import threading
import time
import concurrent.futures as cf
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
ANSWER_RE = re.compile(r"ANSWER:\s*([A-D])")
_lock = threading.Lock()


def render(template, item):
    letters = ",".join(chr(65 + i) for i in range(len(item["choices"])))
    choices = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(item["choices"]))
    return template.format(question=item["question"], choices=choices, letters=letters)


def provider(pin):
    if pin:
        return {"only": [pin], "allow_fallbacks": False, "require_parameters": True, "data_collection": "deny"}
    return {"require_parameters": True, "data_collection": "deny"}


def one(run, item, epoch, temperature, max_tokens, out):
    rec = {"target": run["target"], "label": run["label"], "role": run["role"],
           "sample_id": item["sample_id"], "epoch": epoch}
    prompt = render(run["template"], item)
    for attempt in range(4):  # retry transient rate-limit / network errors
        try:
            r = client.chat.completions.create(
                model=run["model_slug"].removeprefix("openrouter/"),  # direct API drops inspect's prefix
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
                extra_body={"provider": provider(run["pin"])})
            text = r.choices[0].message.content or ""
            m = ANSWER_RE.findall(text)
            rec.update(answer=text, pred=(m[-1] if m else None), target_letter=item["target"],
                       correct=(bool(m) and m[-1] == item["target"]),
                       provider=getattr(r, "provider", None))
            break
        except Exception as e:
            if attempt == 3:
                rec.update(error=str(e))
            else:
                time.sleep(2 * (attempt + 1))
    with _lock:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="payload.json")
    ap.add_argument("--items", default="items.json")
    ap.add_argument("--out", default="answers.jsonl")
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--limit", type=int, help="cap number of tasks (smoke test)")
    a = ap.parse_args()

    p = json.loads(Path(a.payload).read_text())
    items = json.loads(Path(a.items).read_text())

    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["target"], r["label"], r["sample_id"], r["epoch"]))
            except Exception:
                pass

    tasks = [(run, it, ep)
             for run in p["runs"] for it in items for ep in range(p["epochs"])
             if (run["target"], run["label"], it["sample_id"], ep) not in done]
    random.Random(0).shuffle(tasks)  # interleave models so concurrency spreads across all providers at once
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"{len(tasks)} tasks ({len(done)} already done), concurrency {a.concurrency}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(one, run, it, ep, p["temperature"], p["max_tokens"], a.out)
                for run, it, ep in tasks]
        for i, _ in enumerate(cf.as_completed(futs), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
