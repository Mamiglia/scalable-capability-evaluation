"""Load selected verdict-dissection feature blocks without rebuilding them.

The block names come from ``feature-dictionary.parquet``. Current types are:

- surface: readable length, style and formatting counts
- surprisal-llama / surprisal-gemma / surprisal-qwen: frozen-LM predictability
- surprisal-position-adjusted: rolling surprisal corrected for transcript position
- embedding-dense: raw Qwen transcript embedding dimensions
- embedding-sae-shared / embedding-sae-benchmark: transcript SAE activations
- structure: answer presentation and tool-trajectory counts
- clause-topology: structural and similarity-augmented clause graph topology
- clause-node-sae / clause-edge-sae: learned clause and edge activations
- rubric: per-transcript LLM rubric annotations
- exposition: think-aloud, option-matching and labelled-exposition measures
- derivation: equation, justification and answer-derivation measures
- intermediate-errors: quoted and typed slips in correct mathematical solutions
- clause-descriptors: generic graph descriptors
- clause-motifs: named graph and tool-use motifs
- clause-plan: plan presence and execution measures

The dictionary remains the source of truth: a new matrix block can be loaded even
before this explanatory list is updated.
"""
import argparse
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .config import CONFIG

BLOCK_DESCRIPTIONS: dict[str, str] = {
    "surface": "readable length, style and formatting counts",
    "surprisal-llama": "Llama frozen-LM predictability",
    "surprisal-gemma": "Gemma frozen-LM predictability",
    "surprisal-qwen": "Qwen frozen-LM predictability",
    "surprisal-position-adjusted": "rolling surprisal corrected for transcript position",
    "embedding-dense": "raw Qwen transcript embedding dimensions",
    "embedding-sae-shared": "shared transcript SAE activations",
    "embedding-sae-benchmark": "benchmark-specific transcript SAE activations",
    "structure": "answer presentation and tool-trajectory counts",
    "clause-topology": "clause graph topology",
    "clause-node-sae": "clause-node SAE activations",
    "clause-edge-sae": "clause-edge SAE activations",
    "rubric": "per-transcript LLM rubric annotations",
    "exposition": "think-aloud, option-matching and labelled exposition",
    "derivation": "equation, justification and answer derivation",
    "intermediate-errors": "quoted and typed slips in correct solutions",
    "clause-descriptors": "generic graph descriptors",
    "clause-motifs": "named graph and tool-use motifs",
    "clause-plan": "plan presence and execution",
}


Level = Literal["transcript", "pair"]


def _matrix(config: str | Path) -> tuple[Path, pd.DataFrame]:
    """Resolve a design-matrix config to its output root and feature dictionary."""
    config = Path(config)
    values = yaml.safe_load(config.read_text())
    root = Path(values["output_root"])
    dictionary = pd.read_parquet(root / "feature-dictionary.parquet")
    required = {"feature", "block"}
    if missing := required - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    return root, dictionary


def list_blocks(config: str | Path = CONFIG.design_matrix_config) -> pd.DataFrame:
    """Return available block names, feature counts and short descriptions."""
    _, dictionary = _matrix(config)
    blocks = (dictionary.groupby("block", sort=False, as_index=False)
              .agg(n_features=("feature", "size")))
    blocks["description"] = blocks.block.map(BLOCK_DESCRIPTIONS).fillna("not documented")
    return blocks


def load_features(
    level: Level,
    blocks: Sequence[str] | None = None,
    *,
    features: Sequence[str] | None = None,
    splits: str | Sequence[str] | None = None,
    config: str | Path = CONFIG.design_matrix_config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected columns from the existing transcript or pair matrix.

    Pair columns keep their stored ``d_`` prefix; the returned dictionary's
    ``column`` field maps each readable feature name to its stored column. All
    non-feature metadata/target columns are retained. No matrix is rebuilt.

    Args:
        level: ``transcript`` for one row per rollout, or ``pair`` for verdict rows.
        blocks: Feature-dictionary blocks to include; ``None`` loads every block.
        features: Optional exact feature allow-list inside the selected blocks.
        splits: Optional split name(s), such as ``representation_train``.
        config: YAML file whose ``output_root`` contains the design matrices.
    """
    if level not in {"transcript", "pair"}:
        raise ValueError("level must be 'transcript' or 'pair'")
    root, dictionary = _matrix(config)
    available = dictionary.block.drop_duplicates().tolist()
    requested = available if blocks is None else list(dict.fromkeys(blocks))
    unknown = [block for block in requested if block not in available]
    if unknown:
        raise ValueError(f"unknown feature blocks: {', '.join(unknown)}")

    selected = dictionary[dictionary.block.isin(requested)].copy()
    if features:
        feature_order = list(dict.fromkeys(features))
        unknown_features = [feature for feature in feature_order
                            if feature not in set(selected.feature)]
        if unknown_features:
            raise ValueError(f"features not found in selected blocks: {', '.join(unknown_features)}")
        selected = selected.set_index("feature").loc[feature_order].reset_index()
    selected["column"] = selected.feature.map(lambda name: f"d_{name}" if level == "pair" else name)
    path = root / f"{level}.parquet"
    schema = pq.read_schema(path).names
    all_feature_columns = {
        f"d_{feature}" if level == "pair" else feature
        for feature in dictionary.feature
    }
    metadata = [column for column in schema if column not in all_feature_columns]
    missing = [column for column in selected.column if column not in schema]
    if missing:
        raise ValueError(f"matrix missing feature columns: {missing[:5]}")
    table = pd.read_parquet(path, columns=metadata + selected.column.tolist())
    if splits is not None:
        wanted = {splits} if isinstance(splits, str) else set(splits)
        if "split" not in table:
            raise ValueError("matrix has no split column")
        table = table[table.split.isin(wanted)]
    return table.reset_index(drop=True), selected.reset_index(drop=True)


def main() -> None:
    """Print the block inventory or documented feature inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG.design_matrix_config)
    parser.add_argument("--list-blocks", action="store_true")
    parser.add_argument("--list-features", nargs="*", metavar="BLOCK")
    args = parser.parse_args()
    if args.list_blocks:
        print(list_blocks(args.config).to_string(index=False))
    elif args.list_features is not None:
        from .feature_definitions import definitions_for
        _, dictionary = _matrix(args.config)
        if args.list_features:
            dictionary = dictionary[dictionary.block.isin(args.list_features)]
        print(definitions_for(dictionary).to_string(index=False))
    else:
        parser.error("use --list-blocks or --list-features [BLOCK ...]")


if __name__ == "__main__":
    main()
