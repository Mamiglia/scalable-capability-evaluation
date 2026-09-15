"""Read selected correct first attempts into feature rows, without the judging reader.

Only small numeric tables are cached; raw messages are released after extraction.
Source logs are treated as immutable during an evaluation, as in evals.transcripts.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessage

from capability.task import Cohort
from configs.logs import logger
from evals.sample import extract_score
from evals.transcripts import _read_samples
from protocols.transcript_features.features import FEATURES, STRUCTURE_FEATURES, extract_features, transition_features
from reference import tables

ClauseTyper = Callable[[Sequence[ChatMessage]], Sequence[str]]


def read_features(path: str | Path, items: set[str], *, clause_typer: ClauseTyper | None = None) -> pd.DataFrame:
    """Read selected correct first attempts from one Inspect .eval log.

    Args:
        items: Set of item-ID strings to read at epoch 1. Empty means no file access.
        clause_typer: Optional hashable callable taking a sequence of ChatMessage
            objects and returning ordered labels from CLAUSE_TYPES.

    Returns:
        Float DataFrame indexed by item ID (index name: "item"). Columns are
        the 13 non-RST FEATURES (including tool-call count), plus 11 STRUCTURE_FEATURES
        when a labeller is supplied. Incorrect items and absent first attempts
        are omitted. Missing history
        leaves tool-call count and RST/transition values NaN;
        available final-output text can still supply length and regex features.
        The returned table is a copy of the cached result.

    Absent epoch-1 records are reported; missing log files and corrupt records
    still raise their read errors.
    """
    return _read_features(str(path), tuple(sorted(map(str, items))), clause_typer).copy()


@lru_cache(maxsize=128)
def _read_features(path: str, items: tuple[str, ...], clause_typer: ClauseTyper | None) -> pd.DataFrame:
    """Cached implementation of read_features with sorted item IDs as a tuple.

    Returns the stored DataFrame itself; callers must not mutate it. Cache keys
    are (path, items, clause_typer), so file changes do not invalidate a result.
    """
    columns = FEATURES + (STRUCTURE_FEATURES if clause_typer is not None else [])
    rows = {}
    present = ()
    if items:
        with ZipFile(path) as archive:
            members = set(archive.namelist())
        present = tuple(item for item in items if f"samples/{item}_epoch_1.json" in members)
        if len(present) < len(items):
            logger.warning("%s: %d/%d selected items lack epoch 1; using available items",
                           path, len(items) - len(present), len(items))
    # Empty selections need no file access. Never substitute a later correct attempt.
    for sample in _read_samples(path, list(present), epoch=1) if present else ():
        if extract_score(sample):
            rows[str(sample.id)] = sample_features(sample, clause_typer=clause_typer)
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns, dtype=float).rename_axis("item")


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


def feature_table(cohort: Cohort, *, clause_typer: ClauseTyper | None = None) -> pd.DataFrame:
    """Aggregate transcript features into one regression row per model.

    Args:
        cohort: Cohort whose models table contains repeat-0 path_<bench> columns
            and whose items table contains the selected (bench, item) pairs.
            Stored relative paths are resolved under reference.tables.DATA.
        clause_typer: Optional ChatMessage-sequence -> clause-label-sequence
            callable, as in read_features. No capability labels are read here.

    Returns:
        Float DataFrame with the same model index/order as cohort.models and
        FEATURES columns (plus STRUCTURE_FEATURES if requested). Each value is
        the mean of available per-transcript values, weighting items equally
        across benches. Missing paths are skipped; no observations gives NaN.
    """
    columns = FEATURES + (STRUCTURE_FEATURES if clause_typer is not None else [])
    rows = []
    for model in cohort.models.index:
        frames = []
        for bench, item_ids in cohort.items.groupby("bench").item:
            path = cohort.models.loc[model, f"path_{bench}"]
            if isinstance(path, str):
                frames.append(read_features(tables.DATA / path, set(item_ids), clause_typer=clause_typer))
        values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns, dtype=float)
        rows.append(values.mean())
    return pd.DataFrame(rows, index=cohort.models.index, columns=columns, dtype=float)
