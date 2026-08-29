"""Load prepared updated-HF inputs without depending on analysis scripts."""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ANNOTATION_BLOCKS, AnalysisConfig, SOURCE_MANIFEST_SHA256, SOURCE_NAME


FILES = ("feature-dictionary.parquet", "responses.parquet", "transcript.parquet", "pair.parquet")


@dataclass(frozen=True)
class InputTables:
    manifest: dict[str, object]
    dictionary: pd.DataFrame
    variants: dict[str, tuple[str, ...]]
    responses: pd.DataFrame
    transcripts: pd.DataFrame
    pairs: pd.DataFrame


def feature_variants(dictionary: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Return the predeclared feature lists in dictionary order."""
    missing = {"feature", "block"} - set(dictionary)
    if missing:
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    if dictionary.feature.duplicated().any():
        raise ValueError("feature dictionary contains duplicate names")
    features = tuple(dictionary.feature.astype(str))
    no_annotations = tuple(
        row.feature for row in dictionary[["feature", "block"]].itertuples(index=False)
        if row.block not in ANNOTATION_BLOCKS
    )
    return {
        "no_llm_annotations": no_annotations,
        "with_llm_annotations": features,
    }


def validate_manifest(path: str | Path) -> dict[str, object]:
    """Require the exact old annotated readable subset selected by the user."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    source = manifest.get("source", {})
    expected = {"name": SOURCE_NAME, "design_matrix_manifest_sha256": SOURCE_MANIFEST_SHA256}
    if source != expected:
        raise ValueError(
            f"wrong capability input subset in {path}: expected {expected}, found {source}")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(config: AnalysisConfig) -> InputTables:
    """Load one prepared bundle after verifying revisions and file hashes."""
    root = Path(config.input_root)
    manifest = validate_manifest(root / "manifest.json")
    recorded = manifest.get("files", {})
    for name in FILES:
        path = root / name
        expected = recorded.get(name, {}).get("sha256") if isinstance(recorded, dict) else None
        if not path.is_file() or not expected:
            raise ValueError(f"prepared input manifest is missing {name}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")

    dictionary = pd.read_parquet(root / FILES[0])
    variants = feature_variants(dictionary)
    features = list(variants["with_llm_annotations"])
    transcripts = pd.read_parquet(root / "transcript.parquet")
    pairs = pd.read_parquet(root / "pair.parquet")
    if missing := set(features) - set(transcripts):
        raise ValueError(f"transcript table missing features: {sorted(missing)}")
    if missing := {f"d_{feature}" for feature in features} - set(pairs):
        raise ValueError(f"pair table missing features: {sorted(missing)}")
    responses = pd.read_parquet(root / "responses.parquet")
    return InputTables(manifest, dictionary, variants, responses, transcripts, pairs)
