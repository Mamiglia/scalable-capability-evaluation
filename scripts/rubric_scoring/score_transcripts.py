"""Score transcripts against the frozen rubric with a Claude scorer.

The scorer sees one question and one transcript — no pair, no verdicts, no
model identity. Rubric: docs/plans/2026-08-18-transcript-rubric-draft.md
(FROZEN 2026-08-18). Item 15 (surprisal density) is computed separately, not
by the LLM.

Channel: OpenRouter, model anthropic/claude-haiku-4.5 pinned to the
first-party provider, repo-standard safety flags, served provider and
generation id logged per response. Modes: --smoke N; --submit runs the full
set and snapshots the endpoint record alongside the scores.
"""

import argparse
import hashlib
import json
import pathlib

import concurrent.futures
import os
import time

import httpx
import pandas as pd

MODEL = None  # scoring channel NOT approved: user vetoed OpenRouter 2026-08-18
URL = None
# repo-standard OpenRouter safety flags: pin first-party, fail loudly, no data retention
PROVIDER = {"only": ["anthropic"], "allow_fallbacks": False,
            "require_parameters": True, "data_collection": "deny"}

RUBRIC_PROMPT = """You are scoring one transcript of a model answering a multiple-choice science
question. Score only what is in the text. Return ONLY a JSON object, no other text, with
exactly these fields:

- steps (integer): discrete inferences or calculations that advance the solution (an applied
  equation, a case split, an option eliminated). Restated facts and prose padding are not steps.
- slip (boolean): any incorrect intermediate number or equation anywhere in the working, even
  if corrected later or harmless to the final answer. Re-derive the working; do not skim.
- slip_quote (string or null): the offending line, verbatim but at most 20 words, if slip is true.
- conclusion_follows (boolean): the final stated answer is entailed by the immediately
  preceding working.
- self_check (boolean): an explicit verification act (substitution back, unit or
  order-of-magnitude check, re-derivation, "let me verify").
- self_check_type (string or null): which kind, if self_check is true.
- dead_ends (integer): visible abandoned attempts or backtracking ("actually, no", a restarted
  approach, a discarded case).
- backtracking (boolean): dead_ends > 0.
- question_restated (boolean): reformulates or interprets the problem before solving.
- option_strategy (string): "elimination" | "direct" | "mixed".
- letters_without_content (boolean): cites option letters without quoting or using the
  option's content.
- latex_blocks (integer), numbered_lists (integer), headers (integer), bold_spans (integer):
  formatting counts.
- steps_justified (integer 0-2): non-trivial steps carry stated reasons — 0 none, 1 some,
  2 most.
- directness (integer 0-2): 2 shortest-path, 1 minor detours, 0 meandering.
- fact_share (number 0-1): fraction of claims delivered as bare declarative fact with no
  epistemic marking.
- hedges (integer): "I think", "probably", "seems" and similar.
- intensifiers (integer): "clearly", "definitely", "must be" and similar.
- answer_first (boolean): states the answer before the justification.
- repetition (integer): times the conclusion or a key intermediate result is restated beyond
  its first statement.
- option_content_engagement (boolean): quotes or works with actual option text.

Question:
---
{question}
---

Transcript:
---
{transcript}
---"""

SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-moriohamada-PycharmProjects-ai-safety-small-judge/8727e515-2405-4685-a43d-ead3d07c1e1c/scratchpad")
OUT = pathlib.Path(__file__).parent


def transcript_table():
    """Distinct (question, transcript) among the valid balanced selection."""
    scores = pd.read_parquet(SCRATCH / "llama_sentence_scores.parquet")
    selected = set(scores.pair_uid.unique())
    pairs = pd.read_parquet(SCRATCH / "pairs_ceph.parquet")
    pairs = pairs[(pairs.split == "train") & pairs.pair_uid.isin(selected)]
    pairs = pairs.drop_duplicates("pair_uid")
    slots = pd.concat([
        pairs[["question", "model_1", "text_1"]].rename(
            columns={"model_1": "model", "text_1": "text"}),
        pairs[["question", "model_2", "text_2"]].rename(
            columns={"model_2": "model", "text_2": "text"}),
    ]).drop_duplicates(["question", "text"]).reset_index(drop=True)
    slots["tid"] = [hashlib.sha1((q + "\x00" + t).encode()).hexdigest()[:16]
                    for q, t in zip(slots.question, slots.text)]
    return slots.drop_duplicates("tid")


def parse_scores(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    return json.loads(text)


def api_key():
    raise RuntimeError("scoring channel not approved — the user vetoed OpenRouter; "
                       "get an explicit channel + cost approval before any run")


def score_one(client, row, retries=3):
    body = {"model": MODEL, "max_tokens": 2500, "provider": PROVIDER,
            "messages": [{"role": "user", "content": RUBRIC_PROMPT.format(
                question=row.question, transcript=row.text)}]}
    for attempt in range(retries):
        response = client.post(URL, json=body)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        try:
            scores = parse_scores(text)
        except json.JSONDecodeError:
            if attempt + 1 < retries:
                continue
            return {"tid": row.tid, "model": row.model, "error": "unparseable"}
        return {"tid": row.tid, "model": row.model,
                "served_by": data.get("provider"), "generation_id": data.get("id"),
                "tokens_in": data["usage"]["prompt_tokens"],
                "tokens_out": data["usage"]["completion_tokens"], **scores}
    return {"tid": row.tid, "model": row.model, "error": "exhausted retries"}


def run_scoring(slots, out_path, workers=8):
    headers = {"Authorization": f"Bearer {api_key()}"}
    with httpx.Client(headers=headers, timeout=120) as client:
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda r: score_one(client, r),
                                 [row for _, row in slots.iterrows()]))
    table = pd.DataFrame(rows)
    table.to_parquet(out_path, index=False)
    n_err = int(table["error"].notna().sum()) if "error" in table.columns else 0
    print(f"{len(table) - n_err} scored, {n_err} failed -> {out_path}")
    if "served_by" in table.columns:
        print("served by:", table.served_by.value_counts(dropna=False).to_dict())
    return table


def smoke(n):
    slots = transcript_table().sample(n, random_state=0)
    table = run_scoring(slots, OUT / "rubric_scores_smoke.parquet", workers=4)
    total = len(transcript_table())
    if "tokens_in" in table.columns:
        cost = (table.tokens_in.mean() * total * 1e-6 * 1.0
                + table.tokens_out.mean() * total * 1e-6 * 5.0)
        print(f"full-set estimate ({total} transcripts, live price): ${cost:.2f}")
    for _, r in table.iterrows():
        if "steps" in r and pd.notna(r.get("steps")):
            print(r.tid, str(r.model)[:28], "steps", r.steps, "slip", r.slip,
                  "direct", r.directness, "facts", r.fact_share)


def submit():
    slots = transcript_table()
    endpoints = httpx.get(
        "https://openrouter.ai/api/v1/models/anthropic/claude-haiku-4.5/endpoints",
        timeout=30).json()
    (OUT / "endpoints_snapshot.json").write_text(json.dumps(endpoints, indent=1))
    run_scoring(slots, OUT / "rubric_scores.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.smoke)
    elif args.submit:
        submit()


if __name__ == "__main__":
    main()
