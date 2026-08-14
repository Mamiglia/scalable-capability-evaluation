"""Local (vLLM) transcript restyler for the rewrite-augmented SFT set.

Reads rewrite_sources.parquet ({id, model, item, text, target_voice, examples})
and writes rewrites.jsonl (one line per id). Keeps content + reasoning + the
ANSWER line; changes only surface style (neutral, or another model's voice using
`examples`). No OpenRouter — a staged open model on SWC. Resumable: existing ids
are skipped.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

SYSTEM = (
    "You restyle AI answer transcripts for a research dataset. Keep the content "
    "and reasoning IDENTICAL and change ONLY the surface writing style. Preserve "
    "every reasoning step, calculation, intermediate value and claim in the same "
    "order, and keep the final 'ANSWER: <letter>' line verbatim. Never fix or alter "
    "errors. Output only the restyled transcript, nothing else."
)
ANSWER_RE = re.compile(r"ANSWER:\s*([A-D])")


def user_prompt(row):
    if row.target_voice == "neutral":
        instr = ("Rewrite the transcript below in plain, even, neutral prose: strip all "
                 "markdown (headers, bold, bullets, LaTeX) to plain text and remove tone "
                 "and confidence/hedging markers.")
        ex = ""
    else:
        exs = "\n\n---\n".join(list(row.examples))
        instr = (f"Rewrite the transcript below so it reads as if written by the model "
                 f"whose style is shown in these examples. Match their tone, formatting, "
                 f"structure, phrasing, verbosity and confidence. Copy NONE of the examples' "
                 f"content, only their style.\n\n=== STYLE EXAMPLES ===\n{exs}\n=== END EXAMPLES ===")
        ex = ""
    return f"{instr}{ex}\n\n=== TRANSCRIPT TO RESTYLE ===\n{row.text}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N sources")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    df = pd.read_parquet(args.sources).reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]
    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    df = df[~df.id.isin(done)].reset_index(drop=True)

    def render(row):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt(row)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    if args.dry_run:
        print(render(df.iloc[0])[:2000]); print(f"[dry-run] {len(df)} sources to rewrite")
        return

    texts = [render(r) for r in df.itertuples()]
    # drop prompts that leave no room for the rewrite (long transcripts > context)
    cap = args.max_model_len - args.max_new_tokens
    lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    keep = [i for i, l in enumerate(lens) if l <= cap]
    n_drop = len(texts) - len(keep)
    df = df.iloc[keep].reset_index(drop=True)
    texts = [texts[i] for i in keep]
    print(f"dropped {n_drop} over-length prompts (> {cap} tokens); rewriting {len(texts)}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.base_model, max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=args.max_new_tokens, seed=0)
    outs = llm.generate(texts, sp)
    n_ans_ok = 0
    with open(args.out, "a") as f:
        for row, out in zip(df.itertuples(), outs):
            rewrite = out.outputs[0].text.strip()
            src_ans = ANSWER_RE.findall(row.text)
            rw_ans = ANSWER_RE.findall(rewrite)
            ok = bool(rw_ans) and bool(src_ans) and rw_ans[-1] == src_ans[-1]
            n_ans_ok += ok
            f.write(json.dumps(dict(id=row.id, model=row.model, item=str(row.item),
                                    target_voice=row.target_voice, answer_preserved=ok,
                                    rewrite=rewrite)) + "\n")
    print(f"wrote {len(df)} rewrites to {args.out}; answer preserved: {n_ans_ok}/{len(df)}")


if __name__ == "__main__":
    main()
