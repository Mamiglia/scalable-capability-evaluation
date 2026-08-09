"""Generate GPQA answers while copying another model's writing style, as
inspect eval logs.

No baseline: every run copies one source model's style, with the examples
merged into the answering prompt (one user message) so nothing about the
normal task instruction is replaced. For each target model, one run per every
OTHER author model:
  same-tier source  -> control (style matched to the target's own tier)
  other-tier source -> manipulation (weak copies strong = pump-up;
                       strong copies weak = sandbag)

The copy examples are n_examples correct transcripts (final-text only) from
that single source model, drawn only from items OUTSIDE the evaluated set, so
no evaluated item ever appears as an example. Correctness comes from the
choice() scorer, several epochs for headroom.

  --preview   render the prompts (filled with a real item) and exit; no spend
  (default)   run generation

Run (worktree root, main-checkout venv):
  ../small-judge/.venv/bin/python scripts/analyses/style_capability/generate.py --preview
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from judge.util import extract_score, extract_transcript

HERE = Path(__file__).resolve().parent


def provider_args(pin):
    """OpenRouter routing. A pin is a hard endpoint lock (provider + quant, no
    fallback); no pin (proprietary single-served) just denies data collection
    and requires the sampling params be honored."""
    if pin:
        return {"provider": {"only": [pin], "allow_fallbacks": False,
                             "require_parameters": True, "data_collection": "deny"}}
    return {"provider": {"require_parameters": True, "data_collection": "deny"}}


def model_examples(path, exclude_ids, k, seed):
    """k correct final-text transcripts from ONE model's log, one per item, all
    on items outside exclude_ids."""
    uniq = {}
    for s in read_eval_log(path).samples or []:
        if s.id in exclude_ids or s.id in uniq or not extract_score(s):
            continue
        text = extract_transcript(s, include_reasoning=False).strip()
        if text.startswith("[Assistant]: "):  # rendering artifact, not model output
            text = text[len("[Assistant]: "):]
        if text:
            uniq[s.id] = text
    picked = list(uniq.items())
    random.Random(seed).shuffle(picked)
    picked = picked[:k]
    assert all(iid not in exclude_ids for iid, _ in picked), \
        "copy examples must be disjoint from the evaluated items"
    return [t for _, t in picked]


def build_runs(cfg, selected_ids):
    """List of (target, label, role, template). template is a multiple_choice
    template: examples already inserted (braces escaped); {question}/{choices}/
    {letters} left for inspect to fill."""
    g = cfg["generate"]
    authors = cfg["authors"]
    flat = {**authors["strong"], **authors["weak"]}
    tier_of = {m: t for t in ("strong", "weak") for m in authors[t]}

    runs = []
    for target in flat:
        for source in flat:
            if source == target:
                continue
            examples = model_examples(flat[source], selected_ids, g["n_examples"], g["seed"])
            blocks = "\n".join(g["example_template"].format(transcript=t) for t in examples)
            # escape braces from the transcripts (LaTeX) so inspect's str.format
            # of the template does not treat them as fields.
            blocks = blocks.replace("{", "{{").replace("}", "}}")
            template = g["question_template"].replace("{examples}", blocks)
            role = "control" if tier_of[source] == tier_of[target] else "cross"
            runs.append((target, f"copy_{source}", role, template))
    return runs


def build_task(items, target, label, template):
    samples = [
        Sample(id=r.sample_id, input=r.question, choices=list(r.choices), target=r.target)
        for r in items.itertuples()
    ]
    return Task(
        dataset=MemoryDataset(samples),
        solver=[multiple_choice(template=template, shuffle=False)],
        scorer=choice(),
        name=f"gpqa_{target}__{label}",
    )


def render(template, item):
    """Fill a template with one real item, exactly as the model would see it."""
    letters = ",".join(chr(65 + i) for i in range(len(item.choices)))
    choices = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(item.choices))
    return template.format(question=item.question, choices=choices, letters=letters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny check: first run only, 3 items, 1 epoch")
    ap.add_argument("--dump", type=Path,
                    help="write runs + params to JSON for remote execution; no generation")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    g = cfg["generate"]
    out_dir = Path(cfg["out_dir"])
    items = pd.read_parquet(out_dir / "items.parquet")
    runs = build_runs(cfg, set(items.sample_id))

    if args.dump:
        payload = {
            "epochs": g["epochs"], "temperature": g["temperature"], "max_tokens": g["max_tokens"],
            "runs": [{"target": t, "label": l, "role": r, "template": tmpl,
                      "model_slug": g["model_slugs"][t], "pin": g["routing"].get(t)}
                     for t, l, r, tmpl in runs],
        }
        args.dump.write_text(json.dumps(payload))
        print(f"wrote {len(runs)} runs + params -> {args.dump}")
        return 0

    if args.preview:
        prev = out_dir / "prompt_preview"
        prev.mkdir(parents=True, exist_ok=True)
        item0 = items.iloc[0]
        for target, label, role, template in runs:
            (prev / f"{target}__{label}.txt").write_text(render(template, item0))
            print(f"{target:20s} {label:24s} {role:8s} chars={len(template)}")
        control = next((r for r in runs if r[2] == "control"), None)
        cross = next((r for r in runs if r[2] == "cross"), None)
        for tag, r in (("CONTROL (same-tier style)", control), ("CROSS (other-tier style)", cross)):
            print(f"\n===================== {tag}: {r[0]} / {r[1]} =====================\n")
            print(render(r[3], item0))
        print(f"\nwrote {len(runs)} prompts -> {prev}")
        return 0

    sub = "gen"
    if args.smoke:
        runs, items, sub = runs[:1], items.head(3), "gen_smoke"

    failures = []
    for target, label, role, template in runs:
        task = build_task(items, target, label, template)
        try:
            success, _ = eval_set(
                tasks=[task],
                model=g["model_slugs"][target],
                model_args=provider_args(g["routing"].get(target)),
                log_dir=str(out_dir / sub / f"{target}__{label}"),
                epochs=1 if args.smoke else g["epochs"],
                temperature=g["temperature"],
                max_tokens=g["max_tokens"],
                max_connections=40,
            )
            if not success:
                failures.append(f"{target}__{label}")
        except Exception as e:  # one bad run must not kill the rest
            failures.append(f"{target}__{label}: {e}")
        print(f"done {target}__{label} ({role})", flush=True)
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(" ", f)
        return 1
    print("\nall runs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
