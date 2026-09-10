"""Build a domain's artefacts: pooled response table, 1PL difficulties, difficulty quartiles.

    uv run domain configs/domain/science.yml [--warmup 1000] [--samples 1000]

Writes <out_dir>/<domain>/responses.parquet, items.csv (quartile 1 = easiest) and a copy
of the domain config. The fit is a 1PL with a fixed guessing floor per item: 1/n_choices on
multiple-choice items (inspect ``choices``, or an "A. … B. …" list in the question), 0 otherwise.
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_evals.hle.hle import get_verified_gold_uids

from configs.domain import DomainConfig
from configs.logs import logger
from configs.paths import get_eval_path, is_successful_eval
from irt.data import encode
from irt.fitting import fit
from judge.util import extract_score

CHOICE_LINE = re.compile(r"(?m)^\s*([A-Z])[.)]\s")


def guess_rate(sample) -> float:
    """1/n_choices for a multiple-choice item, 0 for free-form."""
    n = len(sample.choices) if sample.choices else len(set(CHOICE_LINE.findall(str(sample.input))))
    return 1.0 / n if n >= 2 else 0.0


def eval_logs(domain: DomainConfig) -> pd.DataFrame:
    """Every successful log of the domain: bench, model, repeat, path (one row per log)."""
    models = domain.model_configs()
    rows = [dict(bench=task.name, model=model.shortname.lower(), repeat=int(log.parent.name), path=log)
            for task in map(domain.task, domain.benches) for model in models
            for log in sorted(get_eval_path(task, model).rglob("*.eval")) if is_successful_eval(log)]
    return pd.DataFrame(rows, columns=["bench", "model", "repeat", "path"])


def responses(domain: DomainConfig) -> pd.DataFrame:
    """One row per (bench, model, item, repeat) from every successful log, after the domain's filters."""
    rows = []
    for bench in domain.benches:
        task = domain.task(bench)
        gold = get_verified_gold_uids() if bench.hle_gold else None
        for model in domain.model_configs():
            for log in sorted(get_eval_path(task, model).rglob("*.eval")):
                if not is_successful_eval(log):
                    continue
                for s in read_eval_log_sample_summaries(str(log)):
                    md = s.metadata or {}
                    if gold is not None and md.get("uid") not in gold:
                        continue
                    if any(md.get(k) not in v for k, v in bench.metadata.items()):
                        continue
                    rows.append(dict(bench=task.name, model=model.shortname.lower(), item=str(s.id),
                                     repeat=int(log.parent.name), correct=bool(extract_score(s)),
                                     guess=guess_rate(s)))
    if not rows:
        raise ValueError(f"domain {domain.name}: no successful logs for any bench x model")
    return pd.DataFrame(rows)


def difficulties(resp: pd.DataFrame, warmup: int = 1000, samples: int = 1000) -> pd.DataFrame:
    """1PL-with-guessing fit over the pooled responses; quartile 1 = easiest quarter of items by b."""
    data = encode(resp.rename(columns={"bench": "benchmark"}))
    guess = resp.groupby(["bench", "item"]).guess.max()  # some imported logs carry no `choices`; the floor is an item property
    guess = np.array([guess[item] for item in data.items])
    post = fit(data, model="1pl", warmup=warmup, samples=samples, guess=guess)
    b = np.asarray(post["b"])
    items = pd.DataFrame(data.items, columns=["bench", "item"]).assign(
        guess=guess, b=b.mean(0), b_lo=np.percentile(b, 5, axis=0), b_hi=np.percentile(b, 95, axis=0))
    items["quartile"] = pd.qcut(items.b.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    return items


def build(domain: DomainConfig, warmup: int = 1000, samples: int = 1000) -> pd.DataFrame:
    resp = responses(domain)
    logger.info("%s: %d responses, %d models, %d items", domain.name, len(resp),
                resp.model.nunique(), len(resp[["bench", "item"]].drop_duplicates()))
    items = difficulties(resp, warmup, samples)
    domain.dir.mkdir(parents=True, exist_ok=True)
    resp.to_parquet(domain.responses_path, index=False)
    items.to_csv(domain.items_path, index=False)
    if domain.source:
        shutil.copy(domain.source, domain.dir / "domain.yml")
    print(items.groupby(["quartile", "bench"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {domain.dir}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    build(DomainConfig.from_cli(args.domain), args.warmup, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
