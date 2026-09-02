"""Run one eval-variation judge config, resumably.

For each comparison [judge, major, minor]: build the judge task, keep only
items whose two transcripts are within max_word_diff words, run, and skip
pairs that already have a success log.

Standard config: transcripts come from one condition's logs (stock judge_task,
post-filtered). Pooled config (pooled: true): each item's transcripts come
from whichever prompt condition pair matches best on length, same-condition
combos preferred on ties.

Usage: run.py CONFIG [--judge SUBSTR] [--model-base-url URL] [--limit N]
"""

import argparse
from functools import lru_cache
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  cluster py3.12 can't read zstd .eval entries without it
except ImportError:
    pass

import yaml
from inspect_ai import Task, eval_set
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

from configs.utils import is_successful_eval
from judge.dataset import responses_to_sample
from judge.scorer import judge_scorer
from judge.task import judge_task
from judge.util import extract_score, extract_transcript

MAX_TRANSCRIPT_CHARS = 65536  # judge_task's max_length default


@lru_cache(maxsize=32)
def first_correct_samples(log_path: str) -> dict:
    """id -> the sample judging would show (first correct epoch)."""
    grouped: dict = {}
    for s in read_eval_log(log_path).samples:
        if extract_score(s):
            grouped.setdefault(s.id, []).append(s)
    return {i: sorted(g, key=lambda s: getattr(s, "epoch", 0))[0] for i, g in grouped.items()}


def words(sample) -> int:
    return len(extract_transcript(sample, include_reasoning=False).split())


def standard_task(cfg: dict, major: str, minor: str) -> Task | None:
    a = first_correct_samples(cfg["transcripts"][major])
    b = first_correct_samples(cfg["transcripts"][minor])
    matched = {i for i in a.keys() & b.keys()
               if abs(words(a[i]) - words(b[i])) <= cfg["max_word_diff"]}
    task = judge_task(
        major_path=cfg["transcripts"][major],
        minor_path=cfg["transcripts"][minor],
        system_prompt=cfg["system_prompt"],
        include_reasoning=cfg["include_reasoning"],
        task_name=f"{major}|{minor}",
    )
    kept = [s for s in task.dataset if s.id.rsplit("_", 1)[0] in matched]
    if not kept:
        return None
    task.dataset = MemoryDataset(kept)
    return task, len(kept)


def pooled_task(cfg: dict, major: str, minor: str) -> Task | None:
    conds = list(cfg["transcripts_by_cond"])
    a = {c: first_correct_samples(cfg["transcripts_by_cond"][c][major]) for c in conds}
    b = {c: first_correct_samples(cfg["transcripts_by_cond"][c][minor]) for c in conds}
    ids = set().union(*(m.keys() for m in a.values())) & set().union(*(m.keys() for m in b.values()))
    samples = []
    for i in sorted(ids, key=str):
        combos = [(abs(words(a[ca][i]) - words(b[cb][i])), ca != cb, ca, cb)
                  for ca in conds for cb in conds if i in a[ca] and i in b[cb]]
        diff, _, ca, cb = min(combos)
        if diff > cfg["max_word_diff"]:
            continue
        sa, sb = a[ca][i], b[cb][i]
        samples.append(responses_to_sample(sa, sb, target="1",
                                           max_len=MAX_TRANSCRIPT_CHARS, include_reasoning=False))
        samples.append(responses_to_sample(sb, sa, target="2",
                                           max_len=MAX_TRANSCRIPT_CHARS, include_reasoning=False))
    if not samples:
        return None
    return Task(
        name=f"{major}|{minor}",
        dataset=MemoryDataset(samples),
        solver=[system_message(cfg["system_prompt"]), generate()],
        scorer=judge_scorer(),
    ), len(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--judge", default=None, help="only run judges whose slug contains this")
    ap.add_argument("--model-base-url", default=None, help="local vLLM endpoint")
    ap.add_argument("--limit", type=int, default=0, help="rows per pair (smoke)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    log_root = Path(cfg["log_root"])
    judge_model_args = cfg.get("judge_model_args") or {}
    build = pooled_task if cfg.get("pooled") else standard_task

    failures = 0
    for judge, major, minor in cfg["comparisons"]:
        if args.judge and args.judge not in judge:
            continue
        judge_short = judge.rsplit("/", 1)[-1]
        pair_dir = log_root / judge_short / f"{major}|{minor}"
        if pair_dir.exists() and any(is_successful_eval(p) for p in pair_dir.glob("*.eval")):
            continue

        built = build(cfg, major, minor)
        if built is None:
            print(f"{judge_short} {major}|{minor}: no length-matched rows, skipped")
            continue
        task, n = built
        print(f"{judge_short} {major}|{minor}: {n} rows")

        kwargs = {"limit": args.limit} if args.limit else {}
        if args.model_base_url:
            kwargs["model_base_url"] = args.model_base_url
        success, _ = eval_set(
            tasks=task,
            model=judge,
            model_args=judge_model_args.get(judge, {}),
            log_dir=str(pair_dir),
            display="plain",
            max_connections=40,
            **kwargs,
        )
        failures += 0 if success else 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
