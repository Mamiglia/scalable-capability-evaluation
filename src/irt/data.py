"""
Turn eval logs into an IRT response table.

Each .eval is one model over the benchmark items. We keep only what IRT needs:
per (model, item, attempt) whether the answer was correct. No transcripts.
Items are keyed by (benchmark, question id), so several evals can share a table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from draccus_kwargs import wrap

SNIPPET_CHARS = 80 # first characters from each question to store to ensure all questions are matched correctly

@dataclass
class DataConfig:
    logs_dir: str = "external_data/epoch-gpqa"
    out: str = "external_data/epoch-gpqa/responses.parquet"
    kwargs: Optional[dict[str, Any]] = None

def question_number(sample) -> str:
    """The question id from the log"""
    return sample.id

def read_log(path: Path) -> Iterator[dict]:
    """
    One .eval to rows: benchmark, model, item, question, correct, answer, target, epoch."""
    log = read_eval_log(str(path))
    benchmark = log.eval.task
    model = path.stem
    for sample in log.samples or []:
        score = next(iter((sample.scores or {}).values()), None)
        if score is None:
            continue
        yield {
            "benchmark": benchmark,
            "model": model,
            "item": question_number(sample),
            "question": " ".join(str(sample.input).split())[:SNIPPET_CHARS],
            "correct": score.value == "C",
            "answer": score.answer,
            "target": sample.target,
            "epoch": sample.epoch,
        }


def build_responses(logs_dir: Path) -> pd.DataFrame:
    """All .eval logs in a dir to one response table."""
    rows = (row for path in sorted(logs_dir.glob("*.eval")) for row in read_log(path))
    return pd.DataFrame(rows)


def items(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per question: benchmark, item, question snippet."""
    return responses[["benchmark", "item", "question"]].drop_duplicates(["benchmark", "item"])


def for_items(responses: pd.DataFrame, item_ids, epoch=None) -> pd.DataFrame:
    """Rows for the given item id(s), all models. Optional single epoch."""
    out = responses[responses["item"].isin(np.atleast_1d(item_ids))]
    return out[out["epoch"] == epoch] if epoch is not None else out


def for_models(responses: pd.DataFrame, model_names, epoch=None) -> pd.DataFrame:
    """Rows for the given model(s), all items. Optional single epoch."""
    out = responses[responses["model"].isin(np.atleast_1d(model_names))]
    return out[out["epoch"] == epoch] if epoch is not None else out


def pick_epoch(responses: pd.DataFrame, epoch: int = 1) -> pd.DataFrame:
    """One row per (model, item): the given epoch, else the next available (wrapping)."""
    order = (responses["epoch"] - epoch) % responses["epoch"].max()
    return (
        responses.assign(_o=order)
        .sort_values("_o")
        .drop_duplicates(["model", "item"])
        .drop(columns="_o")
    )


def verify_alignment(responses: pd.DataFrame) -> dict:
    """Check models share items, and each item number carries one question."""
    items_by_model = responses.groupby("model")["item"].apply(set)
    sets = list(items_by_model)
    union = set().union(*sets) if sets else set()
    shared = set.intersection(*sets) if sets else set()
    questions_per_item = responses.groupby(["benchmark", "item"])["question"].nunique()
    return {
        "n_models": responses["model"].nunique(),
        "n_items_shared": len(shared),
        "n_items_union": len(union),
        "missing_items": {m: len(union - s) for m, s in items_by_model.items() if s != union},
        "clashing_items": questions_per_item[questions_per_item > 1].index.tolist(),
    }


class IRTData(NamedTuple):
    model_idx: np.ndarray
    item_idx: np.ndarray
    y: np.ndarray  # 0/1 correct
    models: list[str]  # model index -> name
    items: list[tuple]  # item index -> (benchmark, question id)


def encode(responses: pd.DataFrame) -> IRTData:
    """Response table to integer-indexed arrays for the fit."""
    pairs = list(zip(responses["benchmark"], responses["item"]))
    models = sorted(responses["model"].unique())
    item_ids = sorted(set(pairs))
    m_index = {m: i for i, m in enumerate(models)}
    i_index = {q: i for i, q in enumerate(item_ids)}
    return IRTData(
        model_idx=responses["model"].map(m_index).to_numpy(),
        item_idx=np.array([i_index[p] for p in pairs]),
        y=responses["correct"].astype(int).to_numpy(),
        models=models,
        items=item_ids,
    )



@wrap
def main(cfg: DataConfig) -> int:
    responses = build_responses(Path(cfg.logs_dir))
    report = verify_alignment(responses)
    print(f"{len(responses)} responses, {report['n_models']} models")
    print(f"items: {report['n_items_shared']} shared / {report['n_items_union']} total")
    for model, missing in report["missing_items"].items():
        print(f"  {model} missing {missing} items")
    for item in report["clashing_items"]:
        print(f"  clashing question for item {item}")
    responses.to_parquet(cfg.out)
    print(f"wrote {cfg.out}")
    return 0


if __name__ == "__main__":
    main()
