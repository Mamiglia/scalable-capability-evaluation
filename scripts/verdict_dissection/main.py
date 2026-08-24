"""Build the design matrix, fit, run ablations.

Design matrix: one row per correct transcript (bench, model, item, epoch)
with the features from features_basic and llm_surprisal, z-scored. Judge
verdicts join later at fit time as pair differences. Built once and cached
as parquet.

  uv run python scripts/verdict_dissection/main.py [--rebuild]
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.scorer import value_to_float

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from judge.util import extract_transcript, strip_think            # noqa: E402
from features_basic import basic_features                          # noqa: E402
from llm_surprisal import CACHE_DIR, SCORERS, cache_file, surprisal_features   # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "verdict_dissection.yaml").read_text())
OUT = ROOT / CFG["data"]["out_dir"]
STORE = OUT / "transcript_store.parquet"      # rendered transcripts, what the judges read
RAW = OUT / "raw_features.parquet"            # features before scaling
DM = OUT / "design_matrix.parquet"            # z-scored features
KEYS = ["bench", "model", "item", "epoch"]


# ---------------------------------------------------------------- transcripts

def canonical_models():
    """Directory names of the evaluated models, from the model yamls:
    the HF-style name after the slash, lowercased (e.g. gpt-5, qwen3.8-27b)."""
    names = set()
    for line in (ROOT / CFG["data"]["models_list"]).read_text().split():
        full = yaml.safe_load((ROOT / line).read_text())["name"]
        names.add(full.split("/", 1)[1].lower())
    return names


def is_correct(sample):
    """Whether the sample scored correct. Scorers store "C"/"I", a float, or
    (HLE) a dict with a "score" entry; anything reaching 1.0 counts."""
    to_float = value_to_float()
    for score in (sample.scores or {}).values():
        value = score.value.get("score", 0) if isinstance(score.value, dict) else score.value
        if to_float(value) >= 1.0:
            return True
    return False


def visible(path):
    """Directory entries that aren't macOS ._ resource forks or other dotfiles."""
    return sorted(p for p in path.iterdir() if not p.name.startswith("."))


def read_rollouts(bench):
    """One row per transcript of one benchmark: bench, model, item, epoch,
    question, transcript (rendered as the judge sees it, no reasoning), correct."""
    root = ROOT / CFG["data"]["rollouts"] / bench / CFG["data"]["collections"][bench]
    models = canonical_models()
    rows = []
    for model_dir in visible(root) if root.exists() else []:          # data/<bench>/<collection>/<model>/
        if model_dir.name.lower() not in models or not model_dir.is_dir():
            continue
        for epoch_dir in visible(model_dir):                           # .../<model>/<epoch>/<one .eval>
            logs = [p for p in visible(epoch_dir) if p.suffix == ".eval"] if epoch_dir.is_dir() else []
            if len(logs) != 1 or read_eval_log(str(logs[0]), header_only=True).status != "success":
                continue
            for sample in read_eval_log_samples(str(logs[0]), all_samples_required=False):   # one sample = one item
                # render exactly what the judge reads: [Assistant]/[Tool] blocks, no reasoning
                text = strip_think(extract_transcript(sample, include_reasoning=False))
                rows.append({"bench": bench, "model": model_dir.name, "item": str(sample.id),
                             "epoch": int(epoch_dir.name), "question": str(sample.input),
                             "transcript": text, "correct": is_correct(sample)})
    return pd.DataFrame(rows)


def build_transcript_store():
    """All benches in the config -> transcript_store.parquet. Prints coverage."""
    store = pd.concat([read_rollouts(b) for b in CFG["data"]["collections"]], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    store.to_parquet(STORE, index=False)
    print(store.groupby("bench").agg(models=("model", "nunique"), transcripts=("item", "size"),
                                     correct=("correct", "mean")).round(3).to_string())
    return store


# ---------------------------------------------------------------- features

def calculate_raw_features(store, surprisal_cache=CACHE_DIR):
    """Feature columns for every transcript in `store`: the basic text features
    always; the surprisal features when the cache has every scorer's file for
    every transcript (otherwise those columns are left out and a note printed)."""
    # surprisal is all-or-nothing: every transcript needs every scorer's .npz
    have_cache = all(cache_file(surprisal_cache, s, r.bench, r.model, r.item, r.epoch).exists()
                     for r in store.itertuples() for s in SCORERS)
    if not have_cache:
        print(f"surprisal cache incomplete under {surprisal_cache}: building without surprisal features")
    rows = []
    for r in store.itertuples():
        feats = basic_features(r.transcript)
        if have_cache:
            feats.update(surprisal_features(r.transcript, r.bench, r.model, r.item, r.epoch, surprisal_cache))
        rows.append({k: getattr(r, k) for k in KEYS} | feats)
    return pd.DataFrame(rows)


def normalise(raw):
    """z-score every feature column over all transcripts. Returns
    (design matrix, scaling table with each column's mean and sd)."""
    features = [c for c in raw.columns if c not in KEYS]
    mean, sd = raw[features].mean(), raw[features].std().replace(0, 1.0)   # sd 0 -> column stays 0, not NaN
    dm = raw.copy()
    dm[features] = (raw[features] - mean) / sd
    scaling = pd.DataFrame({"mean": mean, "sd": sd})
    return dm, scaling


def build_dm(rebuild=False, surprisal_cache=CACHE_DIR):
    """Load design_matrix.parquet, or build it: rollouts -> transcript store ->
    raw features of the correct transcripts -> z-scored design matrix (all
    written to the output dir)."""
    if DM.exists() and not rebuild:
        return pd.read_parquet(DM)
    # slow part: reading ~60k rollout logs; reused unless --rebuild
    store = pd.read_parquet(STORE) if STORE.exists() and not rebuild else build_transcript_store()
    # fast part: features of the correct transcripts (only those ever get judged)
    raw = calculate_raw_features(store[store.correct], surprisal_cache)
    raw.to_parquet(RAW, index=False)
    dm, scaling = normalise(raw)
    dm.to_parquet(DM, index=False)
    scaling.to_csv(OUT / "feature_scaling.csv")
    print(f"design matrix: {len(dm)} transcripts x {len(dm.columns) - len(KEYS)} features -> {DM}")
    return dm


if __name__ == "__main__":
    dm = build_dm(rebuild="--rebuild" in sys.argv)
    print(dm.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
