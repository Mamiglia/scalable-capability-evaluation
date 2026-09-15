"""Per-model feature rows from the per-transcript CSVs in glm_features/ (LASR-G5/glm_features).

scripts/glm_features/extract.py writes one CSV per successful log with extract_log: one row per
sample and reasoning view. feature_table selects a cohort's correct first attempts on its
items from those CSVs and averages per model, so fitting needs no .eval logs at all.
"""

import re
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessage

from capability.task import Cohort
from configs.logs import logger
from evals.sample import extract_score
from protocols.transcript_features.features import FEATURES, STRUCTURE_FEATURES, extract_features, transition_features
from reference import tables

ClauseTyper = Callable[[Sequence[ChatMessage]], Sequence[str]]
MEMBER = re.compile(r"samples/(.+)_epoch_(\d+)\.json$")
VIEWS = ("visible", "hidden")
COLUMNS = FEATURES + STRUCTURE_FEATURES  # structure columns are filled by scripts/glm_features/merge_labels.py


def sample_features(sample: EvalSample, *, include_reasoning: bool = True,
                    clause_typer: ClauseTyper | None = None) -> dict[str, float]:
    """One sample's FEATURES row (plus STRUCTURE_FEATURES with a labeller), regardless of score.

    include_reasoning=False is the reasoning-hidden view: reasoning blocks are dropped
    before every text feature. Without message history the final output still supplies
    text features, but call/structure values are NaN.
    """
    messages = sample.messages
    if not messages:
        choices = getattr(sample.output, "choices", None) or []
        messages = [choices[0].message] if choices else []
    row = extract_features(messages, include_reasoning)
    if not sample.messages:
        row["log_tool_calls"] = np.nan  # a final output cannot establish total trajectory calls
    if clause_typer is not None:
        row.update(transition_features(clause_typer(messages)) if sample.messages
                   else dict.fromkeys(STRUCTURE_FEATURES, np.nan))
    return row


def extract_log(path: str | Path) -> pd.DataFrame:
    """Every sample of one .eval log as rows: item, epoch, correct, reasoning view, FEATURES."""
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not (member := MEMBER.match(name)):
                continue
            sample = EvalSample.model_validate_json(archive.read(name))
            for view in VIEWS:
                rows.append({"item": member[1], "epoch": int(member[2]), "correct": extract_score(sample),
                             "reasoning": view, **sample_features(sample, include_reasoning=view == "visible")})
    return pd.DataFrame(rows, columns=["item", "epoch", "correct", "reasoning", *FEATURES])


def features_path(log_path: str | Path) -> Path:
    """The feature CSV of a data/ log: the same relative path under tables.FEATURES."""
    return tables.FEATURES / Path(log_path).with_suffix(".csv")


def read_features(path: Path, items: set[str], *, reasoning: str = "visible") -> pd.DataFrame:
    """Correct first attempts on `items` from one feature CSV, COLUMNS indexed by item.

    Structure columns a CSV lacks (no RST labels merged yet) read as NaN. A missing CSV
    (log not extracted) is reported and reads as no observations.
    """
    empty = pd.DataFrame(columns=COLUMNS, dtype=float).rename_axis("item")
    if not path.exists():
        logger.warning("%s: no feature CSV; run scripts/glm_features/extract.py", path)
        return empty
    table = pd.read_csv(path, dtype={"item": str})
    keep = (table.epoch == 1) & table.correct & (table.reasoning == reasoning) & table.item.isin(set(map(str, items)))
    return table[keep].set_index("item").reindex(columns=COLUMNS).astype(float) if keep.any() else empty


def feature_table(cohort: Cohort, *, reasoning: str = "visible") -> pd.DataFrame:
    """One COLUMNS row per model of cohort.models: the mean over its available transcripts.

    Reads the repeat-0 path_<bench> logs' CSVs for the cohort's selected (bench, item) pairs,
    weighting items equally across benches. Missing paths are skipped; no observations gives NaN.
    """
    rows = []
    for model in cohort.models.index:
        frames = [read_features(features_path(path), set(item_ids), reasoning=reasoning)
                  for bench, item_ids in cohort.items.groupby("bench").item
                  if isinstance(path := cohort.models.loc[model, f"path_{bench}"], str)]
        rows.append(pd.concat(frames).mean() if frames else pd.Series(np.nan, index=COLUMNS))
    return pd.DataFrame(rows, index=cohort.models.index, columns=COLUMNS, dtype=float)
