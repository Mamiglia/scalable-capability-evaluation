"""Epoch-log member pool: match eval files to ECI, sample gap-stratified pairs."""
import json
import random
import re
from pathlib import Path

from inspect_ai.log import read_eval_log

EFFORT_SUFFIX = re.compile(r"_(low|medium|high|xhigh|none|max|minimal|\d+K)$")

# Epoch API-alias stems -> pool names (only aliases that exist in the pool)
ALIASES = {
    "deepseek-chat": "DeepSeek-V3",
    "deepseek-reasoner": "DeepSeek-R1",
}


def norm(s):
    return re.sub(r"[^a-z0-9. ]", " ", s.lower()).split()


def clean_stem(stem):
    stem = ALIASES.get(stem, stem)
    stem = re.sub(r"-\d{8}$", "", stem)
    stem = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", stem)
    stem = re.sub(r"-(00\d|exp(-\d\d-\d\d)?)$", "", stem)  # gemini API build suffixes
    stem = re.sub(r"^Meta-", "", stem)
    stem = re.sub(r"-(Instruct|instruct)([-_].*)?$", "", stem)
    stem = re.sub(r"-(chat-hf|chat|Chat)$", "", stem)
    return " ".join(norm(stem))


def _dotless(key):
    # version dots vs dashes ("Claude Opus 4.5" vs "claude-opus-4-5")
    return " ".join(key.replace(".", " ").split())


def match_name(pool, stem):
    if EFFORT_SUFFIX.search(stem):
        return None  # effort/budget variants have no clean ECI row
    key = clean_stem(stem)
    for m in pool:
        pool_key = " ".join(norm(m["name"]))
        if pool_key == key or _dotless(pool_key) == _dotless(key):
            return m
    return None


def load_members(bench_dir: Path, pool_json: Path):
    pool = [m for m in json.loads(pool_json.read_text()) if m["eci"] is not None]
    members, unmatched = [], []
    for p in sorted(bench_dir.glob("*.eval")):
        m = match_name(pool, p.stem)
        if m is None:
            unmatched.append(p.stem)
            continue
        members.append({"short": p.stem, "path": p, "eci": m["eci"],
                        "dev": m["dev"], "reasoning": m["reasoning"]})
    print(f"{bench_dir.name}: matched {len(members)}, unmatched {len(unmatched)}")
    return members


def sample_pairs(members, bins, per_bin, seed):
    rng = random.Random(seed)
    pairs = []
    for lo, hi in bins:
        eligible = [(a, b) for a in members for b in members
                    if lo <= a["eci"] - b["eci"] < hi]
        take = min(per_bin, len(eligible))
        if take < per_bin:
            print(f"bin [{lo},{hi}): only {take} of {per_bin} pairs available")
        pairs += rng.sample(eligible, take)
    return pairs


def cot_ok(path: Path) -> bool:
    log = read_eval_log(str(path))
    for s in log.samples[:20]:
        for msg in s.messages:
            if msg.role == "assistant" and isinstance(msg.content, list):
                for c in msg.content:
                    if getattr(c, "type", "") == "reasoning":
                        if getattr(c, "redacted", False) or not (c.reasoning or "").strip():
                            return False
                        return True
    return False  # no reasoning blocks at all


import functools


@functools.lru_cache(maxsize=None)
def _correct_ids(path):
    from judge.util import extract_score
    log = read_eval_log(str(path))
    return {s.id for s in log.samples if extract_score(s)}


def both_correct_ids(path_a, path_b):
    return sorted(_correct_ids(path_a) & _correct_ids(path_b))
