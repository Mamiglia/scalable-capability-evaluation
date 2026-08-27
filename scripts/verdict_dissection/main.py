"""Build the design matrix, fit, run ablations.

Outputs in judge_results/verdict_dissection/ (all pandas parquet):
  transcript_store.parquet  every canonical rollout: bench, model, item, epoch, log,
                            question, transcript (rendered as the judge reads it), correct
  raw_features.parquet      one row per correct transcript: keys + raw feature counts
  design_matrix.parquet     descriptive global z-scores; modelling scales within folds
  verdicts.parquet          one row per judged order: bench, judge, item, model_left,
                            model_right, epoch/log_left, epoch/log_right,
                            judge_log/sample, pick_left
  pairs.parquet             verdicts joined to features, d_<feature> = left - right
Keys: `log` is the rollout file stem (some epoch dirs hold two collection runs).
Built once and reused; --rebuild redoes everything.

  uv run python scripts/verdict_dissection/main.py [--rebuild]
"""
import json
import sys
import hashlib
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from inspect_ai.log import read_eval_log, read_eval_log_samples

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from judge.util import (extract_transcript, extract_transcript_parts, render_transcript_parts,
                        transcript_content_hash, extract_score)   # noqa: E402
from judge.dataset import Transcript, responses_to_sample, has_reasoning   # noqa: E402
from features_basic import basic_features                          # noqa: E402
from llm_surprisal import CACHE_DIR, SCORERS, cache_file, surprisal_features   # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "verdict_dissection.yaml").read_text())
OUT = ROOT / CFG["data"]["out_dir"]
STORE = OUT / "transcript_store.parquet"      # rendered transcripts, what the judges read
RAW = OUT / "raw_features.parquet"            # features before scaling
DM = OUT / "design_matrix.parquet"            # z-scored features
ROLLOUT_INVENTORY = OUT / "rollout_inventory.csv"
ITEM_SPLIT = OUT / "item_split.csv"
RECONCILIATION = OUT / "verdict_reconciliation.json"
FREEZE_MANIFEST = OUT / "freeze_manifest.json"
TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
KEYS = TRANSCRIPT_KEYS + ["sample_epoch", "content_hash"]
CANONICAL_MAX_LENGTH = 262144
LEGACY_USER_PROMPT = """Input question:
---
{question}
---

{options_1}Transcript 1:
---
{transcript_1}
---

{options_2}Transcript 2:
---
{transcript_2}
---

Which of the two transcripts is from the more capable model? Answer in the instructed format.
"""


def select_canonical_logs(candidates):
    """Keep disjoint shards and replace an identical sample-ID set with its later rerun."""
    selected = []
    for name, sample_ids in candidates:
        if not sample_ids:
            raise ValueError(f"canonical rollout log {name} has no samples")
        overlaps = [i for i, (_, ids) in enumerate(selected) if sample_ids & ids]
        if not overlaps:
            selected.append((name, sample_ids))
        elif len(overlaps) == 1 and sample_ids == selected[overlaps[0]][1]:
            selected[overlaps[0]] = (name, sample_ids)
        else:
            raise ValueError(f"canonical rollout logs are partially overlapping at {name}")
    return selected


def canonical_protocol(metadata):
    """Validate the baseline judging protocol; infer only the historical max-length default."""
    expected = {"include_reasoning": False, "truncate_frac": None,
                "require_both_correct": True, "include_reverse": True}
    for key, wanted in expected.items():
        if metadata.get(key, "<missing>") != wanted:
            raise ValueError(f"{key} must be {wanted!r}")
    max_length_recorded = "max_length" in metadata
    max_length = metadata.get("max_length")
    if max_length_recorded and max_length != CANONICAL_MAX_LENGTH:
        raise ValueError(f"max_length must be {CANONICAL_MAX_LENGTH}")
    return expected | {"max_length": max_length, "max_length_recorded": max_length_recorded}


def judge_config_hash(eval_info, metadata, protocol):
    """Hash judge/provider, prompt, generation, scorer, and canonical protocol settings."""
    data = eval_info.model_dump(mode="json")
    prompt_ref = metadata.get("system_prompt")
    prompt_path = ROOT / prompt_ref if prompt_ref else None
    prompt = prompt_path.read_text() if prompt_path and prompt_path.is_file() else prompt_ref
    payload = {key: data.get(key) for key in
               ("model", "model_args", "model_base_url", "model_generate_config", "model_roles", "scorers")}
    payload |= {"system_prompt": prompt, "protocol": protocol}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hash_files(paths):
    """SHA-256 and size for named freeze artifacts."""
    records = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        records[name] = {"path": str(path), "size": Path(path).stat().st_size,
                         "sha256": digest.hexdigest()}
    return records


def dataframe_schema_hash(frame):
    schema = [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def canonical_content_hash(store):
    population = store[store.is_canonical & store.correct][KEYS].sort_values(KEYS)
    return hashlib.sha256(population.to_json(orient="records").encode()).hexdigest()


def freeze_code_sources():
    return {
        "config": ROOT / "configs" / "verdict_dissection.yaml",
        "main_code": Path(__file__),
        "surprisal_code": Path(__file__).with_name("llm_surprisal.py"),
        "judge_util_code": ROOT / "src" / "judge" / "util.py",
    }


def freeze_contract_hash():
    digest = hashlib.sha256()
    for name, path in sorted(freeze_code_sources().items()):
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- transcripts

def canonical_models():
    """Directory names of the evaluated models, from the model yamls:
    the HF-style name after the slash, lowercased (e.g. gpt-5, qwen3.8-27b)."""
    names = set()
    for line in (ROOT / CFG["data"]["models_list"]).read_text().split():
        full = yaml.safe_load((ROOT / line).read_text())["name"]
        names.add(full.split("/", 1)[1].lower())
    return names


def visible(path):
    """Directory entries that aren't macOS ._ resource forks or other dotfiles."""
    return sorted(p for p in path.iterdir() if not p.name.startswith("."))


def read_rollouts(bench):
    """One row per transcript of one benchmark: bench, model, item, epoch,
    question, transcript (rendered as the judge sees it, no reasoning), correct."""
    root = ROOT / CFG["data"]["rollouts"] / bench / CFG["data"]["collections"][bench]
    models = canonical_models()
    canonical_exclusions = set(CFG["data"].get("canonical_log_exclusions", ()))
    candidates = defaultdict(list)
    for model_dir in visible(root) if root.exists() else []:          # data/<bench>/<collection>/<model>/
        model = model_dir.name.lower().removesuffix("_none")          # HF keeps superseded runs under this alias
        if model not in models or not model_dir.is_dir():
            continue
        for epoch_dir in visible(model_dir):                           # .../<model>/<epoch>/*.eval
            logs = [p for p in visible(epoch_dir) if p.suffix == ".eval"] if epoch_dir.is_dir() else []
            for log in logs:                                           # usually one; re-collected models have two
                header = read_eval_log(str(log), header_only=True)
                if header.status != "success":
                    continue
                samples = list(read_eval_log_samples(str(log), all_samples_required=False))
                created = str(getattr(header.eval, "created", "") or "")
                candidates[(model, int(epoch_dir.name))].append((created, log, samples))

    rows = []
    for (model, epoch), logs in sorted(candidates.items()):
        logs.sort(key=lambda candidate: (candidate[0], candidate[1].name, str(candidate[1])))
        by_path = {str(log): (log, samples) for _, log, samples in logs}
        chosen = {path for path, _ in select_canonical_logs([
            (str(log), {str(sample.id) for sample in samples}) for _, log, samples in logs
            if log.stem not in canonical_exclusions
        ])}
        for path, (log, samples) in by_path.items():
            for sample in samples:
                parts = extract_transcript_parts(sample, include_reasoning=False)
                text = render_transcript_parts(parts)
                assert text == extract_transcript(sample, include_reasoning=False)
                question = str(sample.input)
                rows.append({"bench": bench, "model": model, "item": str(sample.id),
                             "epoch": epoch, "log": log.stem,
                             "sample_epoch": int(getattr(sample, "epoch", 0) or 0), "question": question,
                             "transcript": text,
                             "options_json": json.dumps(
                                 list(getattr(sample, "choices", None) or ()), ensure_ascii=False),
                             "transcript_parts_json": json.dumps(parts, ensure_ascii=False),
                             "content_hash": transcript_content_hash(question, text, parts),
                             "correct": extract_score(sample) and has_reasoning(text),   # answer-only outputs are not judged
                             "is_canonical": path in chosen})
    return pd.DataFrame(rows)


def make_rollout_inventory(store):
    return (store.groupby(["bench", "model", "epoch", "log", "is_canonical"])
            .agg(rows=("item", "size"), items=("item", "nunique"),
                 correct_rate=("correct", "mean"), content_hashes=("content_hash", "nunique"))
            .reset_index())


def build_transcript_store(write=True):
    """All benches in the config -> transcript_store.parquet. Prints coverage."""
    store = pd.concat([read_rollouts(b) for b in CFG["data"]["collections"]], ignore_index=True)
    store = (store.sort_values("is_canonical", ascending=False)
             .drop_duplicates(KEYS, keep="first").reset_index(drop=True))
    inventory = make_rollout_inventory(store)
    item_split = make_item_split(store[store.is_canonical], seed=CFG["fit"]["seed"])
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        store.to_parquet(STORE, index=False)
        inventory.to_csv(STORE.parent / ROLLOUT_INVENTORY.name, index=False)
        item_split.to_csv(STORE.parent / ITEM_SPLIT.name, index=False)
    print(store.groupby("bench").agg(models=("model", "nunique"), transcripts=("item", "size"),
                                     correct=("correct", "mean")).round(3).to_string())
    return store


# ---------------------------------------------------------------- features

def calculate_raw_features(store, surprisal_cache=CACHE_DIR):
    """Feature columns for every transcript in `store`: the basic text features
    always; the surprisal features when the cache has every scorer's file for
    every transcript (otherwise those columns are left out and a note printed)."""
    # surprisal is all-or-nothing: every transcript needs every scorer's .npz
    have_cache = all(cache_file(surprisal_cache, s, r.bench, r.model, r.item, r.epoch,
                                r.log, r.content_hash).exists()
                     for r in store.itertuples() for s in SCORERS)
    if not have_cache:
        print(f"surprisal cache incomplete under {surprisal_cache}: building without surprisal features")
    rows = []
    for r in store.itertuples():
        feats = basic_features(r.transcript)
        if have_cache:
            feats.update(surprisal_features(
                r.transcript, json.loads(r.transcript_parts_json), r.bench, r.model,
                r.item, r.epoch, r.log, r.content_hash, surprisal_cache))
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


def scale_fold(train, test, features):
    """Scale train and test data with moments fitted on the training fold only."""
    mean = train[features].mean()
    sd = train[features].std().replace(0, 1.0)
    scaled_train, scaled_test = train.copy(), test.copy()
    scaled_train[features] = (train[features] - mean) / sd
    scaled_test[features] = (test[features] - mean) / sd
    return scaled_train, scaled_test, pd.DataFrame({"mean": mean, "sd": sd})


def make_item_split(rows, seed=0):
    """Seeded 64/16/20 item split, independently shuffled within each benchmark."""
    result = []
    for bench, group in rows.groupby("bench"):
        items = np.array(sorted(group.item.astype(str).unique()))
        np.random.default_rng(seed).shuffle(items)
        n_confirmation, n_validation = round(0.20 * len(items)), round(0.16 * len(items))
        n_train = len(items) - n_confirmation - n_validation
        labels = (["representation_train"] * n_train
                  + ["representation_validation"] * n_validation
                  + ["confirmation"] * n_confirmation)
        result.extend({"bench": bench, "item": item, "split": split}
                      for item, split in zip(items, labels))
    return pd.DataFrame(result).sort_values(["bench", "item"]).reset_index(drop=True)


def build_dm(rebuild=False, surprisal_cache=CACHE_DIR):
    """Load design_matrix.parquet, or build it: rollouts -> transcript store ->
    raw features of the correct transcripts -> z-scored design matrix (all
    written to the output dir)."""
    if DM.exists() and not rebuild:
        return pd.read_parquet(DM)
    # slow part: reading ~60k rollout logs; reused unless --rebuild
    store = pd.read_parquet(STORE) if STORE.exists() and not rebuild else build_transcript_store()
    # fast part: features of the correct transcripts (only those ever get judged)
    raw = calculate_raw_features(store[store.correct & store.is_canonical], surprisal_cache)
    raw.to_parquet(RAW, index=False)
    dm, scaling = normalise(raw)
    dm.to_parquet(DM, index=False)
    scaling.to_csv(OUT / "feature_scaling.csv")
    print(f"design matrix: {len(dm)} transcripts x {len(dm.columns) - len(KEYS)} features -> {DM}")
    return dm


# ---------------------------------------------------------------- verdicts

VERDICTS = OUT / "verdicts.parquet"
PAIRS = OUT / "pairs.parquet"                 # verdicts joined to features: the modelling table
VERDICT_DRAW_KEYS = [
    "bench", "judge", "item", "model_left", "model_right",
    "epoch_left", "epoch_right", "log_left", "log_right", "judge_log", "judge_sample", "judge_epoch",
]


def verdict_draw_key(row):
    """Identity of one judgment draw, including both rollout and judge-log provenance."""
    return tuple(row[k] for k in VERDICT_DRAW_KEYS)


def rollout_key(path):
    """(bench, model, epoch, log stem) from a rollout log path data/<bench>/<collection>/<model>/<epoch>/x.eval."""
    parts = Path(path).parts
    i = parts.index("data") if "data" in parts else len(parts) - 6
    return parts[i + 1], parts[i + 3].lower(), int(parts[i + 4]), Path(path).stem


def visible_deep(root):
    """All .eval files under root, skipping macOS ._ files."""
    return [p for p in root.rglob("*.eval") if not p.name.startswith(".")] if root.exists() else []


def reconcile_verdicts(verdicts, store):
    """Keep only judge logs whose every row reconstructs exactly from the canonical store."""
    retained = store[store.correct].sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    lookup = {tuple(row[key] for key in TRANSCRIPT_KEYS): row
              for row in retained.to_dict("records")}
    accepted, report = [], Counter()
    for _, group in verdicts.groupby(["bench", "judge_log"], sort=False):
        reason, max_length_resolution = None, "recorded"
        for row in group.to_dict("records"):
            left_key = (row["bench"], row["model_left"], row["item"], row["epoch_left"], row["log_left"])
            right_key = (row["bench"], row["model_right"], row["item"], row["epoch_right"], row["log_right"])
            left, right = lookup.get(left_key), lookup.get(right_key)
            if left is None or right is None:
                reason = "unmatched_transcript"
                break
            first = Transcript(row["item"], left["question"], left["transcript"],
                               tuple(json.loads(left["options_json"])), left["sample_epoch"])
            second = Transcript(row["item"], right["question"], right["transcript"],
                                tuple(json.loads(right["options_json"])), right["sample_epoch"])
            if pd.isna(row["max_length"]):
                short = responses_to_sample(first, second, max_len=65536, format_str=LEGACY_USER_PROMPT)
                long = responses_to_sample(first, second, max_len=CANONICAL_MAX_LENGTH,
                                           format_str=LEGACY_USER_PROMPT)
                short_matches = str(short.input) == row["judge_input"]
                long_matches = str(long.input) == row["judge_input"]
                if str(short.input) == str(long.input) and short_matches:
                    rebuilt, max_length_resolution = short, "non_binding"
                elif short_matches and not long_matches:
                    rebuilt, max_length_resolution = short, "identified_65536"
                elif long_matches and not short_matches:
                    rebuilt, max_length_resolution = long, "identified_262144"
                else:
                    reason = "max_length_unresolved"
                    break
            else:
                rebuilt = responses_to_sample(first, second, max_len=int(row["max_length"]))
            if str(rebuilt.input) != row["judge_input"]:
                reason = "input_mismatch"
                break
            recorded = json.loads(row["judge_sample_metadata_json"])
            if any(recorded.get(key) != value for key, value in rebuilt.metadata.items() if key in recorded):
                reason = "metadata_mismatch"
                break
        report[f"{reason}_logs" if reason else "accepted_logs"] += 1
        report[f"{reason}_rows" if reason else "accepted_rows"] += len(group)
        if reason is None:
            group = group.copy()
            group["max_length_resolution"] = max_length_resolution
            accepted.append(group)
    return (pd.concat(accepted, ignore_index=True) if accepted else verdicts.iloc[0:0].copy()), report


def read_verdicts(store=None, write=True):
    """One row per judged order: bench, judge, item, model_left, model_right,
    rollout provenance, judge-log provenance, and pick_left (1 if the judge
    chose the first-shown transcript)."""
    rows, excluded = {}, Counter()
    models = canonical_models()
    for subset in CFG["data"]["judging_subsets"]:
        judging_root = Path(CFG["data"]["judging"]) / subset
        for f in sorted(visible_deep(judging_root)):
            log = read_eval_log(str(f))
            if log.status != "success":
                excluded["log_not_success"] += len(log.samples or [])
                continue
            meta = log.eval.metadata or {}
            judge = log.eval.model.split("/")[-1]
            if judge not in CFG["data"]["judges"]:
                excluded["noncanonical_judge"] += len(log.samples or [])
                continue
            try:
                protocol = canonical_protocol(meta)
                bench, major, epoch_major, log_major = rollout_key(meta["major_path"])
                minor_bench, minor, epoch_minor, log_minor = rollout_key(meta["minor_path"])
            except (KeyError, TypeError, ValueError) as error:
                excluded[f"protocol_or_path:{error}"] += len(log.samples or [])
                continue
            if bench != minor_bench or bench not in CFG["data"]["collections"]:
                excluded["noncanonical_benchmark"] += len(log.samples or [])
                continue
            if major not in models or minor not in models:
                excluded["noncanonical_model"] += len(log.samples or [])
                continue
            config_hash = judge_config_hash(log.eval, meta, protocol)
            for s in log.samples or []:
                try:
                    item, slot = str(s.id).rsplit("_", 1)             # slot = where the major model sat
                except ValueError:
                    excluded["unparseable_sample_id"] += 1
                    continue
                picked = getattr(next(iter((s.scores or {}).values()), None), "answer", None)
                if picked not in ("1", "2"):
                    excluded["unparseable_pick"] += 1
                    continue
                major_first = slot == "1"
                left, right = (major, minor) if major_first else (minor, major)
                ep_left, ep_right = (epoch_major, epoch_minor) if major_first else (epoch_minor, epoch_major)
                log_left, log_right = (log_major, log_minor) if major_first else (log_minor, log_major)
                row = {
                    "bench": bench, "judge": judge, "item": item, "model_left": left, "model_right": right,
                    "epoch_left": ep_left, "epoch_right": ep_right, "log_left": log_left, "log_right": log_right,
                    "judge_log": str(f.relative_to(judging_root)), "judge_sample": str(s.id),
                    "judge_epoch": int(getattr(s, "epoch", 0) or 0),
                    "judge_config_hash": config_hash, "max_length": protocol["max_length"],
                    "max_length_recorded": protocol["max_length_recorded"],
                    "judge_input": str(s.input),
                    "judge_sample_metadata_json": json.dumps(getattr(s, "metadata", None) or {}, sort_keys=True),
                    "pick_left": int(picked == "1")}
                key = verdict_draw_key(row)
                if key in rows:
                    raise ValueError(f"duplicate verdict draw key: {key}")
                rows[key] = row
                excluded["parsed_rows"] += 1
    verdicts = pd.DataFrame(rows.values())
    if store is None:
        if not STORE.exists():
            raise FileNotFoundError(f"canonical transcript store required for verdict reconciliation: {STORE}")
        store = pd.read_parquet(STORE)
    verdicts, reconciliation = reconcile_verdicts(verdicts, store)
    excluded.update(reconciliation)
    verdicts.attrs["reconciliation"] = dict(sorted(excluded.items()))
    if write:
        verdicts.to_parquet(VERDICTS, index=False)
        RECONCILIATION.write_text(json.dumps(verdicts.attrs["reconciliation"], indent=2) + "\n")
    print(verdicts.groupby(["bench", "judge"]).size().unstack(fill_value=0).to_string())
    print("verdict exclusions:", dict(sorted(excluded.items())))
    return verdicts


def pair_rows(verdicts, raw, write=True):
    """Join each verdict to its two transcripts' features: one row per verdict
    with d_<feature> = left minus right. Verdicts whose transcripts are not in
    the design matrix (not correct, or not in the store) are dropped and counted."""
    raw = raw.sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    feats = [c for c in raw.columns if c not in KEYS]
    left = raw.rename(columns={"model": "model_left", "epoch": "epoch_left", "log": "log_left"})
    right = raw.rename(columns={"model": "model_right", "epoch": "epoch_right", "log": "log_right"})
    x = verdicts.merge(left, on=["bench", "item", "model_left", "epoch_left", "log_left"],
                       how="inner", validate="many_to_one")
    x = x.merge(right, on=["bench", "item", "model_right", "epoch_right", "log_right"],
                how="inner", suffixes=("_l", "_r"), validate="many_to_one")
    for f in feats:
        x[f"d_{f}"] = x[f"{f}_l"] - x[f"{f}_r"]
    x = x.drop(columns=[f"{f}_l" for f in feats] + [f"{f}_r" for f in feats])
    if write:
        x.to_parquet(PAIRS, index=False)
    matched = x.groupby("bench").size()
    print("verdicts matched to features, per bench:")
    coverage = pd.DataFrame({"verdicts": verdicts.groupby("bench").size(), "matched": matched})
    print(coverage.fillna(0).astype(int).to_string())
    return x


def write_freeze_manifest(directory, store, verdicts, pairs, final_directory=None):
    """Write hashes and fixed scope for the evidence freeze; caches remain provisional."""
    judge_files = [path for subset in CFG["data"]["judging_subsets"]
                   for path in visible_deep(Path(CFG["data"]["judging"]) / subset)]
    files = {
        "transcript_store": directory / STORE.name,
        "rollout_inventory": directory / ROLLOUT_INVENTORY.name,
        "item_split": directory / ITEM_SPLIT.name,
        "raw_features": directory / RAW.name, "design_matrix": directory / DM.name,
        "feature_scaling": directory / "feature_scaling.csv",
        "verdicts": directory / VERDICTS.name,
        "verdict_reconciliation": directory / RECONCILIATION.name,
        "pairs": directory / PAIRS.name,
        **{name: directory / "code" / path.name for name, path in freeze_code_sources().items()},
    }
    file_hashes = hash_files(files)
    if final_directory is not None:
        for record in file_hashes.values():
            path = Path(record["path"])
            if path.is_relative_to(directory):
                record["path"] = str(final_directory / path.relative_to(directory))
    manifest = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": {
            "command": "configured collection directories and judging_subsets recursive *.eval inventory",
            "rollout_logs": len(make_rollout_inventory(store)), "judge_logs": len(judge_files),
            "judge_cutoff_utc": datetime.fromtimestamp(
                max(path.stat().st_mtime for path in judge_files), timezone.utc).isoformat(),
        },
        "scope": {"benchmarks": list(CFG["data"]["collections"]),
                  "models": sorted(canonical_models()), "judges": list(CFG["data"]["judges"])},
        "row_keys": {"transcript_store": KEYS, "item_split": ["bench", "item", "split"],
                     "verdicts": VERDICT_DRAW_KEYS,
                     "pairs": VERDICT_DRAW_KEYS + ["content_hash_l", "content_hash_r"]},
        "correctness_rule": "judge.util.extract_score and judge.dataset.has_reasoning",
        "population_rule": "is_canonical & correct",
        "transcript_schema_hash": dataframe_schema_hash(store),
        "canonical_content_hash": canonical_content_hash(store),
        "freeze_contract_hash": freeze_contract_hash(),
        "judge_config_hashes": sorted(hash_value for hash_value in verdicts.judge_config_hash.unique()
                                       if isinstance(hash_value, str) and hash_value),
        "protocol": {"include_reasoning": False, "truncate_frac": None,
                     "require_both_correct": True, "include_reverse": True,
                     "recorded_max_length": CANONICAL_MAX_LENGTH,
                     "legacy_max_length_bounds": [65536, CANONICAL_MAX_LENGTH]},
        "split": {"seed": CFG["fit"]["seed"], "representation_train": 0.64,
                  "representation_validation": 0.16, "confirmation": 0.20},
        "rows": {"store": len(store), "canonical_store": int(store.is_canonical.sum()),
                 "canonical_correct": int((store.is_canonical & store.correct).sum()),
                 "verdicts": len(verdicts), "pairs": len(pairs)},
        "caches": {"status": "provisional_not_frozen"},
        "transcript_store": file_hashes["transcript_store"],
        "item_split": file_hashes["item_split"],
        "files": file_hashes,
    }
    if not manifest["judge_config_hashes"]:
        raise ValueError("freeze requires nonempty judge_config_hashes")
    manifest_path = directory / FREEZE_MANIFEST.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def rebuild_and_freeze():
    """Compute every Stage A artifact before atomically promoting the freeze."""
    store = build_transcript_store(write=False)
    inventory = make_rollout_inventory(store)
    item_split = make_item_split(store[store.is_canonical], seed=CFG["fit"]["seed"])
    raw = calculate_raw_features(store[store.correct & store.is_canonical])
    dm, scaling = normalise(raw)
    verdicts = read_verdicts(store=store, write=False)
    pairs = pair_rows(verdicts, raw, write=False)

    freeze_root = OUT / "freezes"
    freeze_root.mkdir(parents=True, exist_ok=True)
    target = freeze_root / f"v1-{canonical_content_hash(store)[:12]}-{freeze_contract_hash()[:12]}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {target}")
    stage = Path(tempfile.mkdtemp(prefix=".stage-a-", dir=freeze_root))
    store.to_parquet(stage / STORE.name, index=False)
    inventory.to_csv(stage / ROLLOUT_INVENTORY.name, index=False)
    item_split.to_csv(stage / ITEM_SPLIT.name, index=False)
    raw.to_parquet(stage / RAW.name, index=False)
    dm.to_parquet(stage / DM.name, index=False)
    scaling.to_csv(stage / "feature_scaling.csv")
    verdicts.to_parquet(stage / VERDICTS.name, index=False)
    (stage / RECONCILIATION.name).write_text(
        json.dumps(verdicts.attrs["reconciliation"], indent=2) + "\n")
    pairs.to_parquet(stage / PAIRS.name, index=False)
    code_dir = stage / "code"
    code_dir.mkdir()
    for source in freeze_code_sources().values():
        shutil.copy2(source, code_dir / source.name)
    write_freeze_manifest(stage, store, verdicts, pairs, final_directory=target)
    stage.rename(target)
    return target / FREEZE_MANIFEST.name


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        print(f"freeze -> {rebuild_and_freeze()}")
        raise SystemExit
    build_dm()
    verdicts = pd.read_parquet(VERDICTS) if VERDICTS.exists() else read_verdicts()
    pairs = pair_rows(verdicts, pd.read_parquet(RAW))
    print(pairs.groupby("judge").pick_left.agg(["mean", "size"]).round(3).to_string())
