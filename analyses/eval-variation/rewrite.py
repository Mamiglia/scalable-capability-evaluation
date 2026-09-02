"""Rewrite baseline gpqa transcripts to ~200 words, post-hoc.

Two modes: self (each model rewrites its own first-correct-epoch transcripts)
and standard (one rewriter model does all of them). Output is a resumable
jsonl; convert_rewrites.py turns it into judge-readable eval logs.

Usage:
  rewrite.py --mode self [--limit 10]
  rewrite.py --mode standard --rewriter gemini-3.1-flash-lite [--limit 10]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from gen_configs import CHEAP7_CONFIG, CONDITIONS, SUBJECTS_BY_ECI, resolve_log  # noqa: E402
from run import first_correct_samples  # noqa: E402
from judge.util import extract_transcript  # noqa: E402

PROMPT = (
    "Rewrite the following answer to a multiple-choice question so it is "
    "approximately 200 words long. Preserve the reasoning, its content and "
    "its conclusions. Keep the final line exactly as in the original "
    "(\"ANSWER: X\"). Change only wording and length.\n\n{original}"
)
ANSWER_RE = re.compile(r"ANSWER\s*:\s*[*_$\s]*\(?([A-Za-z])\)?", re.IGNORECASE)


def openrouter_client() -> OpenAI:
    key = None
    for line in open(".env"):
        if line.strip().startswith("OPENROUTER_API_KEY"):
            key = line.split("=", 1)[1].strip()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("self", "standard"), required=True)
    ap.add_argument("--rewriter", default="gemini-3.1-flash-lite",
                    help="standard mode: shortname of the single rewriter")
    ap.add_argument("--limit", type=int, default=0, help="items per model (smoke)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or Path(f"judge_results/eval_prompt_variation/rewrites_{args.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["model"], r["item"]))

    pins = yaml.safe_load(Path(CHEAP7_CONFIG).read_text())["judge_model_args"]
    pins = {k.removeprefix("openrouter/"): v for k, v in pins.items()}
    # 2026-08-12: alibaba's qwen3-8b listing dropped its fp8 quant label, so the
    # alibaba/fp8 tag stopped matching. Same endpoint, provider-only pin.
    pins["qwen/qwen3-8b"] = {"provider": {"only": ["alibaba"], "allow_fallbacks": False,
                                          "require_parameters": True, "data_collection": "deny"}}
    slugs = {  # shortname -> openrouter slug
        "qwen3-8b": "qwen/qwen3-8b", "qwen3-32b": "qwen/qwen3-32b",
        "qwen3-vl-30b-a3b-instruct": "qwen/qwen3-vl-30b-a3b-instruct",
        "glm-4.6": "z-ai/glm-4.6", "ling-2.6-flash": "inclusionai/ling-2.6-flash",
        "deepseek-v4-flash-0731": "deepseek/deepseek-v4-flash-0731",
        "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite",
    }
    client = openrouter_client()
    fh = open(out, "a")

    def rewrite_one(model, item, original, rewriter_short):
        slug = slugs[rewriter_short]
        body = {"provider": pins[slug]["provider"], "reasoning": {"enabled": False}}
        for attempt in (body, {k: v for k, v in body.items() if k != "reasoning"}):
            try:
                resp = client.chat.completions.create(
                    model=slug, temperature=0.2, max_tokens=1200,
                    messages=[{"role": "user", "content": PROMPT.format(original=original)}],
                    extra_body=attempt,
                )
                text = resp.choices[0].message.content or ""
                if not text.strip():
                    continue  # empty completion: retry without reasoning knob
                m = ANSWER_RE.search(text)
                return {"model": model, "item": str(item), "rewriter": rewriter_short,
                        "original_words": len(original.split()),
                        "rewritten_words": len(text.split()),
                        "new_answer": m.group(1).upper() if m else None,
                        "rewritten": text}
            except Exception as e:
                err = e
        return {"model": model, "item": str(item), "rewriter": rewriter_short,
                "error": str(err)[:200]}

    jobs = []
    for m, _ in SUBJECTS_BY_ECI:
        samples = first_correct_samples(resolve_log(CONDITIONS["baseline"], m))
        items = sorted(samples, key=str)[: args.limit or None]
        rewriter = m if args.mode == "self" else args.rewriter
        for i in items:
            if (m, str(i)) in done:
                continue
            original = extract_transcript(samples[i], include_reasoning=False)
            jobs.append((m, i, original, rewriter))

    print(f"{len(jobs)} rewrites to do ({len(done)} already present)")
    ok = err = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        for res in ex.map(lambda j: rewrite_one(*j), jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            ok += "rewritten" in res
            err += "error" in res
    print(f"done: {ok} ok, {err} errors -> {out}")


if __name__ == "__main__":
    main()
