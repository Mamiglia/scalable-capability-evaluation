"""Pairwise rubric annotation: an LLM sees both transcripts of a judged pair and says, per named
dimension, which one is better (1, 2 or 0 for a tie). Never asked for an overall verdict.

    uv run python scripts/verdict_dissection/annotate_pairs.py --model gpt-5-mini --openrouter [--benches gpqa aime] [--limit N]
    uv run python scripts/verdict_dissection/annotate_pairs.py --model gpt-5-mini --export

Pairs are the distinct unordered rollout pairs of the judge GLM's rows; each is shown once, in an
order fixed by a hash of the pair, so the annotator's position bias averages out over pairs.
Results append to judge_results/verdict_dissection/annotations/pairs-<model>.jsonl (resumable).
--export writes <features_root>/pairwise/pairwise-<model>.parquet: one row per shown order of every
verdict pair with the signed comparison per dimension (+1 = the left-shown transcript is better).
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from prepare import PAIR_IDENTITY  # noqa: E402
from build_design_matrices import pool_aime  # noqa: E402

OUT = ROOT / "judge_results/verdict_dissection/annotations"
TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
SIDE_COLUMNS = ["model", "epoch", "log"]
MAX_TRANSCRIPT_CHARS = 60_000

DIMENSIONS = {
    "more_rigorous": "which transcript justifies its steps more carefully",
    "more_efficient": "which transcript reaches the solution more directly, with fewer detours or redundant steps",
    "fewer_errors": "which transcript has fewer incorrect statements, wrong calculations or invalid inferences on the way (both final answers are correct)",
    "deeper_understanding": "which transcript shows more genuine understanding of the problem beyond the minimum needed",
    "clearer": "which transcript is easier to follow",
    "better_verified": "which transcript checks its own result more convincingly (tests, substitution, re-derivation, sanity checks)",
    "more_complete": "which transcript addresses more of what the question asks, including why alternatives are rejected",
    "better_tool_use": "agentic transcripts only, else 0: which transcript uses commands and tools more competently (fewer failed, wasted or repeated actions)",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(DIMENSIONS),
          "properties": {k: {"type": "integer", "description": f"1, 2 or 0 (tie): {v}", "enum": [0, 1, 2]} for k, v in DIMENSIONS.items()}}
SYSTEM = ("You compare two AI models' transcripts answering the same question. Both reach the correct final answer; do not judge "
          "the answer, compare the working. For each dimension say 1 if Transcript 1 is better, 2 if Transcript 2 is better, "
          "0 if they are about equal. Return only the JSON object described by the schema.")
OPENROUTER_PROVIDER = {"gpt-5-mini": "openai", "gpt-5": "openai", "claude-haiku-4.5": "anthropic"}


def cut(text, budget=MAX_TRANSCRIPT_CHARS):
    return text if len(text) <= budget else text[:budget * 2 // 3] + "\n[... middle omitted ...]\n" + text[-budget // 3:]


def unordered_pairs(verdict_pairs):
    """Distinct rollout pairs with the two sides in a canonical order plus the order to show them in."""
    key = verdict_pairs[PAIR_IDENTITY].drop_duplicates().copy()
    swap = key.model_left > key.model_right
    for column in SIDE_COLUMNS:
        left, right = key.loc[swap, f"{column}_left"].copy(), key.loc[swap, f"{column}_right"].copy()
        key.loc[swap, f"{column}_left"], key.loc[swap, f"{column}_right"] = right.to_numpy(), left.to_numpy()
    key = key.drop_duplicates(PAIR_IDENTITY).reset_index(drop=True)
    digest = key[PAIR_IDENTITY].astype(str).agg("|".join, axis=1).map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    key["canonical_first"] = digest.map(lambda h: int(h[0], 16) % 2 == 0)      # shown as Transcript 1 when True
    return key


def with_text(pairs, transcripts):
    for side in ("left", "right"):
        keys = {k: f"{k}_{side}" if k in SIDE_COLUMNS else k for k in TRANSCRIPT_KEYS}
        table = transcripts.rename(columns={**keys, "transcript": f"text_{side}", "options_json": f"options_{side}"})
        if side == "right":
            table = table.drop(columns="question")
        pairs = pairs.merge(table, on=list(keys.values()), how="left", validate="many_to_one")
    if pairs.text_left.isna().any() or pairs.text_right.isna().any():
        raise ValueError("a pair has no transcript text")
    return pairs


def options_block(options_json, n):
    options = json.loads(options_json) if options_json else []
    if not options:
        return ""
    return f"\nOptions as shown to Model {n}:\n" + "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options)) + "\n"


def user_message(row):
    first, second = ("left", "right") if row.canonical_first else ("right", "left")
    return (f"Question:\n{row.question}\n"
            f"{options_block(getattr(row, f'options_{first}'), 1)}\nTranscript 1:\n{cut(getattr(row, f'text_{first}'))}\n\n"
            f"{options_block(getattr(row, f'options_{second}'), 2)}\nTranscript 2:\n{cut(getattr(row, f'text_{second}'))}")


async def annotate(client, model, row, semaphore, extra_body):
    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model, extra_body=extra_body,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(row)}],
                    response_format={"type": "json_schema", "json_schema": {"name": "comparison", "strict": True, "schema": SCHEMA}})
                usage = response.usage
                return {**{k: getattr(row, k) for k in PAIR_IDENTITY}, "canonical_first": bool(row.canonical_first),
                        **json.loads(response.choices[0].message.content),
                        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                        "served_by": (response.model_extra or {}).get("provider")}
            except Exception as error:
                if attempt == 4:
                    return {**{k: getattr(row, k) for k in PAIR_IDENTITY}, "error": repr(error)[:300]}
                await asyncio.sleep(2 ** attempt)


async def run(rows, model, concurrency, out_path, openrouter, effort):
    if openrouter:
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        provider = OPENROUTER_PROVIDER[model]
        model, extra_body = f"{provider}/{model}", {"provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": True},
                                                    "reasoning": {"effort": effort}}
    else:
        client, extra_body = AsyncOpenAI(), None
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [annotate(client, model, row, semaphore, extra_body) for row in rows.itertuples(index=False)]
    done, t0 = 0, time.time()
    with out_path.open("a") as out:
        for coroutine in asyncio.as_completed(tasks):
            out.write(json.dumps(await coroutine) + "\n")
            out.flush()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} in {(time.time() - t0) / 60:.1f} min", flush=True)


def export(model, config, effort="low"):
    """Signed comparisons for every shown order of every verdict pair: +1 = left-shown transcript better.
    With several effort files present, the comparisons are averaged over them (a second pass reduces annotator noise)."""
    files = sorted(OUT.glob(f"pairs-{model}*.jsonl"))
    rows = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    rows = rows[rows.error.isna()] if "error" in rows else rows
    for name in DIMENSIONS:
        towards_first = rows[name].map({1: 1, 2: -1, 0: 0})                       # +1 = Transcript 1 better
        rows[name] = towards_first * rows.canonical_first.map({True: 1, False: -1})   # +1 = canonical left better
    signed = rows.groupby(PAIR_IDENTITY, as_index=False)[list(DIMENSIONS)].mean()    # mean over passes
    print(f"{len(files)} annotation files, {len(rows)} ratings, {len(signed)} pairs")
    both = [signed]
    flipped = signed.copy()
    for column in SIDE_COLUMNS:
        flipped[f"{column}_left"], flipped[f"{column}_right"] = signed[f"{column}_right"], signed[f"{column}_left"]
    for name in DIMENSIONS:
        flipped[name] = -signed[name]
    both.append(flipped)
    table = pd.concat(both, ignore_index=True)
    out = Path(config["features_root"]) / "pairwise"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / f"pairwise-{model}.parquet", index=False)
    print(f"{len(signed)} annotated pairs -> {len(table)} shown orders -> pairwise/pairwise-{model}.parquet")
    print(signed.groupby("bench")[list(DIMENSIONS)].apply(lambda d: (d != 0).mean()).round(2).T.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--benches", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--effort", default="low", help="reasoning effort (OpenRouter); a non-low effort writes to its own file")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/design_matrices.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.export:
        export(args.model, config)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / (f"pairs-{args.model}.jsonl" if args.effort == "low" else f"pairs-{args.model}-{args.effort}.jsonl")
    verdict_pairs = pd.read_parquet(Path(config["output_root"]) / "pair.parquet", columns=PAIR_IDENTITY)
    pairs = unordered_pairs(verdict_pairs)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet",
                            columns=TRANSCRIPT_KEYS + ["sample_epoch", "question", "options_json", "transcript", "correct", "is_canonical"])
    store = pool_aime(store[store.is_canonical & store.correct]).sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    pairs = with_text(pairs, store[TRANSCRIPT_KEYS + ["question", "options_json", "transcript"]])
    if args.benches:
        pairs = pairs[pairs.bench.isin(args.benches)]
    if out_path.exists():
        done = pd.read_json(out_path, lines=True)
        done = done[done.error.isna()] if "error" in done else done
        pairs = pairs.merge(done[PAIR_IDENTITY].drop_duplicates(), on=PAIR_IDENTITY, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"{len(pairs)} pairs to annotate with {args.model}", flush=True)
    asyncio.run(run(pairs, args.model, args.concurrency, out_path, args.openrouter, args.effort))
    results = pd.read_json(out_path, lines=True)
    if "error" in results:
        print("errors:", int(results.error.notna().sum()), results.error.dropna().head(2).tolist())
    if "prompt_tokens" in results:
        print(f"tokens so far: prompt {results.prompt_tokens.sum() / 1e6:.2f}M, completion {results.completion_tokens.sum() / 1e6:.2f}M")


if __name__ == "__main__":
    main()
