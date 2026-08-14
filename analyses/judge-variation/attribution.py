"""Family attribution: can a judge tell which model family wrote a transcript?

One transcript per prompt, fixed multiple-choice family list, final text only.
Chance = 1/6. Never mixed into the comparison conditions.

Usage:
  uv run python attribution.py --bench gpqa \
      --judge-model openai-api/local/judge --judge-short qwen3-0.6b [--dry-run]
"""
import argparse
import random
import re
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  epoch .eval zips use zstd; py<3.14 zipfile needs this shim
except ImportError:
    pass

from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import (
    Score, Target, CORRECT, INCORRECT, NOANSWER, accuracy, stderr, scorer,
)
from inspect_ai.solver import generate, system_message

from judge.util import extract_transcript
from pool import load_members

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

FAMILIES = {"Alibaba (Qwen)": "A", "Meta": "B", "DeepSeek": "C",
            "OpenAI": "D", "Anthropic": "E", "Google": "F"}

SYSTEM = """You are shown one AI model's response to a question. Identify which model family wrote it.

A) Qwen (Alibaba)
B) Llama (Meta)
C) DeepSeek
D) GPT (OpenAI)
E) Claude (Anthropic)
F) Gemini (Google)

You MUST follow this reply format:

EXPLANATION: <What in the style or content points to a family. 1-3 sentences.>
ANSWER: <One letter A-F.>
CONFIDENCE: <An integer between 1 and 5: 1 = guessing, 5 = certain.>
"""

USER = """Input question:
---
{question}
---

Response:
---
{transcript}
---

Which model family wrote this response? Answer in the instructed format.
"""


@scorer(metrics=[accuracy(), stderr()])
def family_scorer():
    async def score(state, target: Target):
        text = state.output.completion if state.output else ""
        m = re.search(r"ANSWER:[*_]*\s*[*_$]*([A-F])", text, re.IGNORECASE)
        if not m:
            return Score(value=NOANSWER, answer=None)
        picked = m.group(1).upper()
        return Score(value=CORRECT if picked == target.text else INCORRECT, answer=picked)
    return score


def attribution_task(bench_dir, pool_json, per_family=40, seed=0, max_length=65536):
    members = [m for m in load_members(bench_dir, pool_json) if m["dev"] in FAMILIES]
    rng = random.Random(seed)
    samples = []
    for fam_dev in FAMILIES:
        fam_members = [m for m in members if m["dev"] == fam_dev]
        picks = []
        for m in fam_members:
            log = read_eval_log(str(m["path"]))
            for s in log.samples:
                picks.append((m, s))
        if not picks:
            print(f"attribution: no members for {fam_dev}")
            continue
        for k, (m, s) in enumerate(rng.sample(picks, min(per_family, len(picks)))):
            transcript = extract_transcript(s, include_reasoning=False)[:max_length]
            samples.append(Sample(
                # epoch logs repeat item ids across epochs; k keeps ids unique
                id=f"{m['short']}__{s.id}__{k}",
                input=USER.format(question=s.input, transcript=transcript),
                target=FAMILIES[fam_dev],
                metadata={"member": m["short"], "dev": fam_dev},
            ))
    print(f"attribution: {len(samples)} samples")
    return Task(name=f"attribution_{bench_dir.name}", dataset=MemoryDataset(samples),
                solver=[system_message(SYSTEM), generate()], scorer=family_scorer())


def has_success_log(directory: Path) -> bool:
    for p in directory.glob("*.eval"):
        try:
            if read_eval_log(str(p), header_only=True).status == "success":
                return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-short", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-connections", type=int, default=32)
    ap.add_argument("--log-root", default="", help="override log dir (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_dir = Path(args.log_root) if args.log_root else \
        ROOT / f"logs/judging-variations/{args.bench}/attribution/{args.judge_short}"
    if has_success_log(log_dir):
        print("already judged, nothing to do")
        return
    task = attribution_task(ROOT / "external_data" / f"epoch-{args.bench}",
                            ROOT / "model-sampler" / "model_pool.json",
                            per_family=args.per_family, seed=args.seed)
    if args.dry_run:
        return
    success, _ = eval_set(
        tasks=[task], model=args.judge_model, model_base_url=args.base_url,
        log_dir=str(log_dir), max_connections=args.max_connections,
    )
    if not success:
        raise SystemExit("FAILED: attribution")


if __name__ == "__main__":
    main()
