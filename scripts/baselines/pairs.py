"""Rows shared by the stronger-model baselines: one row per distinct transcript pair with the
stronger side by benchmark accuracy over all canonical rollouts (AIME years pooled), the assistant
text of every transcript, the held-out designs of glm/prepare.py, and the held_out_*.csv scores
in glm/evaluate.py's format. Baselines write <output_dir>/stronger_model_from_pair/<name>/<design>/
so glm/collate.py and glm/compare_to_judges.py read them like GLM waves.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
from build_design_matrices import pool_aime          # noqa: E402
from features_basic import _assistant_text          # noqa: E402
from prepare import PAIR_IDENTITY, designs, held_out  # noqa: E402,F401

MODEL = "stronger_model_from_pair"
TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]


def load_config():
    return yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())


def load_identity(config, columns):
    """Canonical rollouts with the AIME years pooled, as in the design matrices."""
    identity = pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet", columns=columns + ["is_canonical"])
    return pool_aime(identity[identity.is_canonical].drop(columns="is_canonical"))


def load_pairs(config):
    """One row per distinct transcript pair; left_stronger by the models' accuracy over all canonical rollouts."""
    pairs = pd.read_parquet(Path(config["output_root"]) / "pair.parquet", columns=PAIR_IDENTITY).drop_duplicates()
    accuracy = (load_identity(config, ["bench", "model", "item", "correct"]).groupby(["bench", "model"], as_index=False)
                .correct.mean().rename(columns={"correct": "accuracy"}))
    for side in ("left", "right"):
        pairs = pairs.merge(accuracy.rename(columns={"model": f"model_{side}", "accuracy": f"accuracy_{side}"}),
                            on=["bench", f"model_{side}"], how="left", validate="many_to_one")
    if pairs[["accuracy_left", "accuracy_right"]].isna().any().any():
        raise ValueError("a model is missing its benchmark accuracy")
    pairs = pairs[pairs.accuracy_left != pairs.accuracy_right]
    return pairs.assign(left_stronger=(pairs.accuracy_left > pairs.accuracy_right).astype(int)).reset_index(drop=True)


def load_transcripts(config):
    """Assistant text of every canonical transcript, one per transcript key."""
    identity = load_identity(config, TRANSCRIPT_KEYS + ["sample_epoch", "transcript"])
    identity = identity.sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    return (identity[TRANSCRIPT_KEYS].assign(text=identity.transcript.map(_assistant_text))
            .reset_index(drop=True))


def transcript_rows(pairs, transcripts, side):
    """Row of `transcripts` holding each pair's left or right transcript."""
    keys = {k: f"{k}_{side}" if k in ("model", "epoch", "log") else k for k in TRANSCRIPT_KEYS}
    index = transcripts[TRANSCRIPT_KEYS].reset_index().rename(columns=keys)
    matched = pairs.merge(index, on=list(keys.values()), how="left", validate="many_to_one")
    if matched["index"].isna().any():
        raise ValueError(f"{int(matched['index'].isna().sum())} pairs have no {side} transcript")
    return matched["index"].to_numpy(int)


def report(predictions):
    """Accuracy and log loss overall and per benchmark, as glm/evaluate.py writes them."""
    rows = []
    for name, group in [("all", predictions)] + list(predictions.groupby("bench")):
        p, y = group.predicted.clip(1e-6, 1 - 1e-6), group.observed
        rows.append({"group": name, "n": len(group), "accuracy": float(((p > 0.5) == y).mean()),
                     "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())})
    return pd.DataFrame(rows)


def write_design(output_dir, name, design, predictions):
    """predictions_*.csv and held_out_*.csv for one baseline and design; returns the scores."""
    design_dir = output_dir / MODEL / name / design
    design_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(design_dir / f"predictions_{MODEL}.csv", index=False)
    scores = report(predictions)
    scores.to_csv(design_dir / f"held_out_{MODEL}.csv", index=False)
    return scores
