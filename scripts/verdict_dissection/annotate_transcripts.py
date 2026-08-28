"""Rubric annotation of every canonical-correct transcript by an LLM, as transcript features.

    uv run python scripts/verdict_dissection/annotate_transcripts.py --model gpt-5-mini [--benches gpqa aime2024 ...] [--limit N]

One call per transcript (never the pair), JSON output with the counts and 1-5
ratings in RUBRIC. Calls go to the OpenAI API (OPENAI_API_KEY) or, with --openrouter, to OpenRouter
(OPENROUTER_API_KEY) pinned to the model's first-party provider so no quantised or truncated
serving slips in. Results append to
judge_results/verdict_dissection/annotations/<model>.jsonl keyed by the transcript keys and
content_hash, so a rerun skips what is done. --limit N annotates the first N pending
transcripts (a cost check). Afterwards --export writes
<features_root>/raw/rubric-<model>.parquet in the feature-table layout.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
OUT = ROOT / "judge_results/verdict_dissection/annotations"
MAX_TRANSCRIPT_CHARS = 60_000          # head and tail kept beyond this (SWE-bench trajectories)

RUBRIC = {
    "reasoning_errors": "integer: incorrect statements, wrong calculations or invalid inferences in the working (the final answer is known to be correct; count mistakes on the way)",
    "unjustified_claims": "integer: key claims or steps asserted without justification",
    "redundant_steps": "integer: steps, checks or actions that repeat earlier work or do not contribute to the solution",
    "self_corrections": "integer: places where the model notices and corrects its own mistake",
    "verifies_result": "0 or 1: checks its own result (substitution, test, re-derivation, sanity check)",
    "explains_reasoning": "0 or 1: gives a rationale rather than only an answer or only actions",
    "considers_alternatives": "0 or 1: discusses other options, approaches or hypotheses and why they are rejected",
    "wasted_actions": "integer (agentic transcripts only, else 0): commands or tool calls that failed, were unnecessary or led nowhere",
    "clarity": "1-5: how easy the reasoning is to follow",
    "rigour": "1-5: how carefully each step is justified",
    "efficiency": "1-5: how directly the solution is reached, without detours",
    "depth": "1-5: how much genuine understanding of the problem is shown beyond the minimum needed",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(RUBRIC),
          "properties": {k: {"type": "integer", "description": v} for k, v in RUBRIC.items()}}
SYSTEM = ("You annotate one AI model's transcript for a question it answered correctly. Do not judge the final answer; "
          "rate the working. Return only the JSON object described by the schema. Be literal and consistent: counts are counts, "
          "ratings use the whole 1-5 range.")


def cut(text, budget=MAX_TRANSCRIPT_CHARS):
    if len(text) <= budget:
        return text
    return text[:budget * 2 // 3] + "\n[... middle omitted ...]\n" + text[-budget // 3:]


def user_message(row):
    options = json.loads(row.options_json) if row.options_json else []
    options_block = ("\nOptions:\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options)) + "\n") if options else ""
    return f"Question:\n{row.question}\n{options_block}\nTranscript:\n{cut(row.transcript)}"


async def annotate(client, model, row, semaphore, extra_body):
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model, extra_body=extra_body,          # gpt-5 models reject a temperature; default sampling
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(row)}],
                    response_format={"type": "json_schema", "json_schema": {"name": "rubric", "strict": True, "schema": SCHEMA}})
                usage = response.usage
                served = (response.model_extra or {}).get("provider")          # OpenRouter reports who served the call
                return {**{k: getattr(row, k) for k in KEYS}, **json.loads(response.choices[0].message.content),
                        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                        "served_by": served, "served_model": response.model}
            except Exception as error:      # rate limits and transient failures: back off, then give up on the row
                if attempt == 4:
                    return {**{k: getattr(row, k) for k in KEYS}, "error": repr(error)[:300]}
                await asyncio.sleep(2 ** attempt)


OPENROUTER_PROVIDER = {"gpt-5-mini": "openai", "gpt-5": "openai", "claude-haiku-4.5": "anthropic"}


async def run(rows, model, concurrency, out_path, openrouter):
    if openrouter:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        provider = OPENROUTER_PROVIDER[model]
        model = f"{provider}/{model}"
        extra_body = {"provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": True},
                      "reasoning": {"effort": "low"}}          # gpt-5 reasoning at default effort costs ~1.3k tokens a call
    else:
        client, extra_body = AsyncOpenAI(), None
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [annotate(client, model, row, semaphore, extra_body) for row in rows.itertuples(index=False)]
    done, t0 = 0, time.time()
    with out_path.open("a") as out:
        for coroutine in asyncio.as_completed(tasks):
            result = await coroutine
            out.write(json.dumps(result) + "\n")
            out.flush()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} in {(time.time() - t0) / 60:.1f} min", flush=True)


def export(model, features_root):
    """Feature table from the annotations: one row per transcript in the surface table, missing rows NaN."""
    rows = pd.read_json(OUT / f"{model}.jsonl", lines=True)
    rows = rows[rows.get("error").isna()] if "error" in rows else rows
    rows = rows.drop_duplicates(KEYS, keep="last")
    keys = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS)
    table = keys.merge(rows[KEYS + list(RUBRIC)], on=KEYS, how="left", validate="one_to_one")
    table.to_parquet(features_root / f"raw/rubric-{model}.parquet", index=False)
    print(f"{int(table[list(RUBRIC)[0]].notna().sum())}/{len(table)} transcripts annotated -> raw/rubric-{model}.parquet")
    print(table.groupby("bench")[list(RUBRIC)].mean().round(2).T.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--benches", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--openrouter", action="store_true", help="call through OpenRouter pinned to the first-party provider")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    if args.export:
        export(args.model, features_root)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.model}.jsonl"
    keys = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet", columns=KEYS + ["question", "options_json", "transcript"])
    rows = keys.merge(store, on=KEYS, how="left", validate="one_to_one")
    if args.benches:
        rows = rows[rows.bench.isin(args.benches)]
    if out_path.exists():
        done = pd.read_json(out_path, lines=True)
        done = done[done.get("error").isna()] if "error" in done else done
        rows = rows.merge(done[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    if args.limit:
        rows = rows.head(args.limit)
    print(f"{len(rows)} transcripts to annotate with {args.model}; ~{(rows.transcript.str.len().clip(upper=MAX_TRANSCRIPT_CHARS).sum() + rows.question.str.len().sum()) / 3.8 / 1e6:.1f}M input tokens", flush=True)
    asyncio.run(run(rows, args.model, args.concurrency, out_path, args.openrouter))
    results = pd.read_json(out_path, lines=True)
    if "error" in results:
        print("errors:", int(results.error.notna().sum()), results.error.dropna().head(2).tolist())
    if "prompt_tokens" in results:
        print(f"tokens so far: prompt {results.prompt_tokens.sum() / 1e6:.2f}M, completion {results.completion_tokens.sum() / 1e6:.2f}M")


if __name__ == "__main__":
    main()
