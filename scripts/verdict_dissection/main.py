"""Build the design matrix, fit, run ablations.

Outputs in judge_results/verdict_dissection/ (all pandas parquet):
  transcript_store.parquet  every canonical rollout: bench, model, item, epoch, log,
                            question, transcript (rendered as the judge reads it), correct
  raw_features.parquet      one row per correct transcript: keys + raw feature counts
  design_matrix.parquet     the same, z-scored (feature_scaling.csv has mean/sd)
  verdicts.parquet          one row per judged order: bench, judge, item, model_left,
                            model_right, epoch/log_left, epoch/log_right, pick_left
  pairs.parquet             verdicts joined to features, d_<feature> = left - right
Keys: `log` is the rollout file stem (some epoch dirs hold two collection runs).
Built once and reused; --rebuild redoes everything.

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
KEYS = ["bench", "model", "item", "epoch", "log"]   # log = rollout file stem; some epoch dirs hold two runs


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
        model = model_dir.name.lower().removesuffix("_none")          # HF keeps superseded runs under this alias
        if model not in models or not model_dir.is_dir():
            continue
        for epoch_dir in visible(model_dir):                           # .../<model>/<epoch>/*.eval
            logs = [p for p in visible(epoch_dir) if p.suffix == ".eval"] if epoch_dir.is_dir() else []
            for log in logs:                                           # usually one; re-collected models have two
                if read_eval_log(str(log), header_only=True).status != "success":
                    continue
                for sample in read_eval_log_samples(str(log), all_samples_required=False):   # one sample = one item
                    # render exactly what the judge reads: [Assistant]/[Tool] blocks, no reasoning
                    text = strip_think(extract_transcript(sample, include_reasoning=False))
                    rows.append({"bench": bench, "model": model, "item": str(sample.id),
                                 "epoch": int(epoch_dir.name), "log": log.stem, "question": str(sample.input),
                                 "transcript": text, "correct": is_correct(sample)})
    return pd.DataFrame(rows)


def build_transcript_store():
    """All benches in the config -> transcript_store.parquet. Prints coverage."""
    store = pd.concat([read_rollouts(b) for b in CFG["data"]["collections"]], ignore_index=True).drop_duplicates(ignore_index=True)
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
    have_cache = all(cache_file(surprisal_cache, s, r.bench, r.model, r.item, r.epoch, r.log).exists()
                     for r in store.itertuples() for s in SCORERS)
    if not have_cache:
        print(f"surprisal cache incomplete under {surprisal_cache}: building without surprisal features")
    rows = []
    for r in store.itertuples():
        feats = basic_features(r.transcript)
        if have_cache:
            feats.update(surprisal_features(r.transcript, r.bench, r.model, r.item, r.epoch, r.log, surprisal_cache))
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


# ---------------------------------------------------------------- verdicts

VERDICTS = OUT / "verdicts.parquet"
PAIRS = OUT / "pairs.parquet"                 # verdicts joined to features: the modelling table


def rollout_key(path):
    """(bench, model, epoch, log stem) from a rollout log path data/<bench>/<collection>/<model>/<epoch>/x.eval."""
    parts = Path(path).parts
    i = parts.index("data") if "data" in parts else len(parts) - 6
    return parts[i + 1], parts[i + 3].lower(), int(parts[i + 4]), Path(path).stem


def visible_deep(root):
    """All .eval files under root, skipping macOS ._ files."""
    return [p for p in root.rglob("*.eval") if not p.name.startswith(".")] if root.exists() else []


def read_verdicts():
    """One row per judged order: bench, judge, item, model_left, model_right,
    epoch_left, epoch_right, pick_left (1 if the judge chose the first-shown
    transcript). Later logs of the same (judge, pair, item, order) win."""
    rows = {}
    for subset in CFG["data"]["judging_subsets"]:
        for f in sorted(visible_deep(Path(CFG["data"]["judging"]) / subset)):
            log = read_eval_log(str(f))
            if log.status != "success":
                continue
            meta = log.eval.metadata or {}
            judge = log.eval.model.split("/")[-1]
            if judge not in CFG["data"]["judges"]:
                continue                                               # e.g. Phi-3 pilot, not in the roster
            bench, major, epoch_major, log_major = rollout_key(meta["major_path"])
            _, minor, epoch_minor, log_minor = rollout_key(meta["minor_path"])
            for s in log.samples or []:
                item, slot = str(s.id).rsplit("_", 1)             # slot = where the major model sat
                picked = getattr(next(iter((s.scores or {}).values()), None), "answer", None)
                if picked not in ("1", "2"):
                    continue                                       # judge gave no usable verdict
                major_first = slot == "1"
                left, right = (major, minor) if major_first else (minor, major)
                ep_left, ep_right = (epoch_major, epoch_minor) if major_first else (epoch_minor, epoch_major)
                log_left, log_right = (log_major, log_minor) if major_first else (log_minor, log_major)
                rows[(bench, judge, item, left, right)] = {
                    "bench": bench, "judge": judge, "item": item, "model_left": left, "model_right": right,
                    "epoch_left": ep_left, "epoch_right": ep_right, "log_left": log_left, "log_right": log_right,
                    "pick_left": int(picked == "1")}
    verdicts = pd.DataFrame(rows.values())
    verdicts.to_parquet(VERDICTS, index=False)
    print(verdicts.groupby(["bench", "judge"]).size().unstack(fill_value=0).to_string())
    return verdicts


def pair_rows(verdicts, dm):
    """Join each verdict to its two transcripts' features: one row per verdict
    with d_<feature> = left minus right. Verdicts whose transcripts are not in
    the design matrix (not correct, or not in the store) are dropped and counted."""
    feats = [c for c in dm.columns if c not in KEYS]
    left = dm.rename(columns={"model": "model_left", "epoch": "epoch_left", "log": "log_left"})
    right = dm.rename(columns={"model": "model_right", "epoch": "epoch_right", "log": "log_right"})
    x = verdicts.merge(left, on=["bench", "item", "model_left", "epoch_left", "log_left"], how="inner")
    x = x.merge(right, on=["bench", "item", "model_right", "epoch_right", "log_right"], how="inner", suffixes=("_l", "_r"))
    for f in feats:
        x[f"d_{f}"] = x[f"{f}_l"] - x[f"{f}_r"]
    x = x.drop(columns=[f"{f}_l" for f in feats] + [f"{f}_r" for f in feats])
    x.to_parquet(PAIRS, index=False)
    matched = x.groupby("bench").size()
    print("verdicts matched to features, per bench:")
    print(pd.DataFrame({"verdicts": verdicts.groupby("bench").size(), "matched": matched}).fillna(0).astype(int).to_string())
    return x


if __name__ == "__main__":
    dm = build_dm(rebuild="--rebuild" in sys.argv)
    verdicts = pd.read_parquet(VERDICTS) if VERDICTS.exists() and "--rebuild" not in sys.argv else read_verdicts()
    pairs = pair_rows(verdicts, dm)
    print(pairs.groupby("judge").pick_left.agg(["mean", "size"]).round(3).to_string())
