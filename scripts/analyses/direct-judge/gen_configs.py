# scripts/analyses/direct-judge/gen_configs.py
"""Generate direct-judge configs from the pairwise baseline configs.

The transcript set is the union over pairs: every (model, item) shown in any
baseline pairwise comparison, listed per model. Pair lists and per-pair item
ids are copied through verbatim for analysis-time induced picks, and each
model's ECI is embedded so analysis never needs pool matching.

The inputs live on the judging-variations branch; materialize them first:
  git show judging-variations:analyses/judge-variation/configs/gpqa_a_baseline.yaml > <dir>/gpqa_a_baseline.yaml
  git show judging-variations:model-sampler/model_pool.json > <dir>/model_pool.json

Usage:
  uv run python gen_configs.py --pairwise-config <dir>/gpqa_a_baseline.yaml \
      --pool-json <dir>/model_pool.json --bench gpqa
"""
import argparse
import json
import re
from pathlib import Path

import yaml

from score_task import SYSTEM_PROMPT

HERE = Path(__file__).parent

# -- model short (eval file stem) -> pool entry, same matching as the pairwise
#    pool sampler (analyses/judge-variation/pool.py on judging-variations) --

EFFORT_SUFFIX = re.compile(r"_(low|medium|high|xhigh|none|max|minimal|\d+K)$")

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
    stem = re.sub(r"-(00\d|exp(-\d\d-\d\d)?)$", "", stem)
    stem = re.sub(r"^Meta-", "", stem)
    stem = re.sub(r"-(Instruct|instruct)([-_].*)?$", "", stem)
    stem = re.sub(r"-(chat-hf|chat|Chat)$", "", stem)
    return " ".join(norm(stem))


def _dotless(key):
    return " ".join(key.replace(".", " ").split())


def match_name(pool, stem):
    if EFFORT_SUFFIX.search(stem):
        return None
    key = clean_stem(stem)
    for m in pool:
        pool_key = " ".join(norm(m["name"]))
        if pool_key == key or _dotless(pool_key) == _dotless(key):
            return m
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairwise-config", type=Path, required=True)
    ap.add_argument("--pool-json", type=Path, required=True)
    ap.add_argument("--bench", required=True)
    args = ap.parse_args()

    src = yaml.safe_load(args.pairwise_config.read_text())
    pool = [m for m in json.loads(args.pool_json.read_text()) if m["eci"] is not None]

    ids_by_model: dict = {}
    for major, minor in src["comparisons"]:
        for short in (major, minor):
            ids_by_model.setdefault(short, set()).update(
                src["sample_ids"][f"{major}-vs-{minor}"])

    eci = {}
    for short in ids_by_model:
        m = match_name(pool, short)
        assert m is not None, f"no ECI match for {short}"
        eci[short] = m["eci"]

    cfg = dict(
        log_root=f"logs/direct-judge/{args.bench}",
        bench=args.bench,
        system_prompt=SYSTEM_PROMPT,
        max_length=src["max_length"],
        include_reasoning=src["include_reasoning"],
        transcripts={s: src["transcripts"][s] for s in sorted(ids_by_model)},
        sample_ids={s: sorted(ids_by_model[s], key=str) for s in sorted(ids_by_model)},
        eci=eci,
        comparisons=src["comparisons"],
        pair_sample_ids=src["sample_ids"],
    )
    out_dir = HERE / "configs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.bench}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100000))
    n_transcripts = sum(len(v) for v in cfg["sample_ids"].values())
    print(f"{args.bench}: {len(ids_by_model)} models, {n_transcripts} transcripts, "
          f"{len(cfg['comparisons'])} pairs -> {out}")


if __name__ == "__main__":
    main()
