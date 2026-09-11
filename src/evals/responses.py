"""The response table of a domain, read from its collected logs.

    responses: one row per (bench, model, item, repeat) with `correct` and `guess` (the item's
               guessing floor: 1/n_choices on multiple choice — inspect `choices` or an
               "A. … B. …" list in the question — else 0), after the domain's item filters.
    eval_logs: one row per successful log: bench, model, repeat, path.
"""

import re
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_evals.hle.hle import get_verified_gold_uids

from configs.domain import DomainConfig
from configs.paths import get_eval_path, is_successful_eval
from evals.sample import extract_score

CHOICE_LINE = re.compile(r"(?m)^\s*([A-Z])[.)]\s")


def guess_rate(sample) -> float:
    """1/n_choices for a multiple-choice item, 0 for free-form."""
    if sample.choices:
        n = len(sample.choices)
    else:
        n = len(set(CHOICE_LINE.findall(str(sample.input))))
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


def repeat0_paths(logs: pd.DataFrame, relative_to: Path) -> pd.DataFrame:
    """One column per bench, `path_<bench>`: each model's repeat-0 log, relative to `relative_to`.

    Protocols read transcripts from these logs. NaN where the model was never run on the bench.
    """
    repeat0 = logs[logs.repeat == 0].copy()
    duplicates = repeat0.groupby(["bench", "model"]).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates):
        raise ValueError(f"several repeat-0 logs for {duplicates.index.tolist()}")
    repeat0["path"] = [str(Path(p).relative_to(relative_to)) for p in repeat0.path]
    return repeat0.pivot(index="model", columns="bench", values="path").add_prefix("path_")
