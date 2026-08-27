"""Build the transcript and pair design matrices for the verdict-dissection GLMs.

Feature blocks come from configs/design_matrices.yaml. Extensive (size-like) non-negative
features are log(1 + x) transformed, so their pair differences are log ratios; everything
else is differenced as is. The three AIME years are one benchmark, "aime", with items
prefixed by their year. Columns are z-scaled per benchmark on representation_train items
(pairs: scaled by the difference SD, not centred, so reversed pairs stay negated). Missing
values become 0 after scaling. Targets: model_accuracy (transcript rows) and pick_left,
accuracy_difference, left_stronger (pair rows), with model accuracies computed on
development items leaving the row's own item out.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from judge.dataset import has_reasoning   # noqa: E402
TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
KEYS = TRANSCRIPT_KEYS + ["sample_epoch", "content_hash"]
PAIR_KEYS = ["bench", "judge", "item", "model_left", "model_right", "epoch_left", "epoch_right",
             "log_left", "log_right", "judge_log", "judge_sample", "judge_epoch", "pick_left"]
STRUCTURE_COUNTS = ("assistant_turns|tool_calls|tool_errors|tool_outputs_empty|submit_calls|test_commands|repeated_tool_calls"
                    "|distinct_tools|display_equations|inline_math|option_references|elimination_terms|bold_spans|headers")
EXTENSIVE = re.compile(rf"^(?!.*gain)(n_words|.*_count|.*_chars|.*_sum|surprisal_(full|noq|local)|{STRUCTURE_COUNTS})(__\w+)?$")
DEVELOPMENT = {"representation_train", "representation_validation"}


def pool_aime(frame):
    """One benchmark for the AIME years: bench "aime", item "<year>/<item>" so ids stay unique."""
    frame = frame.copy()
    aime = frame.bench.str.startswith("aime")
    frame.loc[aime, "item"] = frame.bench[aime] + "/" + frame.item[aime].astype(str)
    frame.loc[aime, "bench"] = "aime"
    return frame


def block_columns(frame, pattern):
    return [c for c in frame.columns if c not in KEYS and re.search(pattern, c)]


def transform(frame, features):
    """log1p on extensive non-negative features; returns the frame and a transform table."""
    out, rows = frame.copy(), []
    for f in features:
        extensive = bool(EXTENSIVE.match(f)) and f != "log_chars" and frame[f].min() >= 0
        if extensive:
            out[f] = np.log1p(frame[f].astype(float))
        rows.append({"feature": f, "transform": "log1p" if extensive else "none"})
    return out, pd.DataFrame(rows)


def leave_item_out_accuracy(identity, split):
    """Per (bench, model, item): the model's solve rate on development items other than this one."""
    rows = identity.loc[identity.is_canonical, ["bench", "model", "item", "correct"]]
    rows = rows.merge(split, on=["bench", "item"], how="left", validate="many_to_one")
    dev = rows[rows.split.isin(DEVELOPMENT)]
    totals = dev.groupby(["bench", "model"]).correct.agg(["sum", "size"])
    own = dev.groupby(["bench", "model", "item"]).correct.agg(["sum", "size"])
    out = rows[["bench", "model", "item"]].drop_duplicates().set_index(["bench", "model", "item"])
    out = out.join(totals, on=["bench", "model"]).join(own, rsuffix="_own").fillna(0)
    out["model_accuracy"] = (out["sum"] - out["sum_own"]) / (out["size"] - out["size_own"])
    return out.reset_index()[["bench", "model", "item", "model_accuracy"]]


def drop_answer_only(frame, pairs, store):
    """Drop transcripts with no reasoning text (has_reasoning) and every pair that shows one.

    The v1 freeze marked answer-only outputs correct; later freezes apply the rule in the store.
    """
    bad = store.loc[~store.transcript.map(has_reasoning), TRANSCRIPT_KEYS].drop_duplicates()
    bad["answer_only"] = True
    frame = frame.merge(bad, on=TRANSCRIPT_KEYS, how="left")
    frame = frame[frame.answer_only.isna()].drop(columns="answer_only")
    for side in ("left", "right"):
        keys = {k: f"{k}_{side}" if k not in ("bench", "item") else k for k in TRANSCRIPT_KEYS}
        pairs = pairs.merge(bad.rename(columns=keys), on=list(keys.values()), how="left")
        pairs = pairs[pairs.answer_only.isna()].drop(columns="answer_only")
    return frame.reset_index(drop=True), pairs.reset_index(drop=True), len(bad)


def pair_differences(pairs, transcripts, features):
    """Left minus right feature values, matched on the transcript keys, as d_ columns."""
    base = (transcripts.sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
            .reset_index(drop=True))
    index = base[TRANSCRIPT_KEYS].reset_index()
    sides = {}
    for side in ("left", "right"):
        keys = {k: f"{k}_{side}" if k != "bench" and k != "item" else k for k in TRANSCRIPT_KEYS}
        matched = pairs.merge(index.rename(columns=keys), on=list(keys.values()), how="left",
                              validate="many_to_one")
        if matched["index"].isna().any():
            raise ValueError(f"{side} transcripts missing for {matched['index'].isna().sum()} pairs")
        sides[side] = matched["index"].to_numpy(int)
    values = base[features].to_numpy(np.float32)
    return pd.DataFrame(values[sides["left"]] - values[sides["right"]],
                        columns=[f"d_{f}" for f in features], index=pairs.index)


def scale(frame, features, centre):
    """z-scale per benchmark on representation_train rows; NaN -> 0 afterwards."""
    values = frame[features].to_numpy(np.float32, copy=True)
    train = (frame.split == "representation_train").to_numpy()
    rows = []
    for bench in pd.unique(frame.bench):
        mask = (frame.bench == bench).to_numpy()
        reference = values[mask & train]
        mean = np.nanmean(reference, axis=0) if centre else np.zeros(len(features), np.float32)
        sd = np.nanstd(reference, axis=0, ddof=1)
        sd = np.where(np.isnan(sd) | (sd == 0), 1.0, sd)
        values[mask] = (values[mask] - mean) / sd
        rows.extend({"benchmark": bench, "feature": f, "mean": float(m), "sd": float(s)}
                    for f, m, s in zip(features, mean, sd))
    out = frame.drop(columns=features)
    out = pd.concat([out, pd.DataFrame(np.nan_to_num(values), columns=features, index=frame.index)], axis=1)
    return out[frame.columns], pd.DataFrame(rows)


def load_blocks(cfg):
    """One transcript-level frame with every block's columns, and the block dictionary."""
    tables, frame, rows = {}, None, []
    for block in cfg["blocks"]:
        path = ROOT / block["table"].format(features_root=cfg["features_root"])
        if path not in tables:
            tables[path] = pd.read_parquet(path)
        table = tables[path]
        columns = block_columns(table, block["columns"])
        part = table[KEYS + columns]
        if "fill_missing" in block:
            part = part.fillna({c: block["fill_missing"] for c in columns})
        frame = part if frame is None else frame.merge(part, on=KEYS, how="inner", validate="one_to_one")
        rows.extend({"feature": c, "block": block["name"], "source": str(path.relative_to(ROOT))
                     if path.is_relative_to(ROOT) else str(path)} for c in columns)
    return frame, pd.DataFrame(rows)


def build(cfg):
    freeze = ROOT / cfg["freeze"]
    features_root = Path(cfg["features_root"])
    split = pool_aime(pd.read_csv(freeze / "item_split.csv"))
    identity = pool_aime(pd.read_parquet(features_root / "raw/identity.parquet",
                                         columns=["bench", "model", "item", "correct", "is_canonical"]))
    frame, dictionary = load_blocks(cfg)          # blocks join on the original keys, then AIME is pooled
    features = dictionary.feature.tolist()
    frame, transforms = transform(frame, features)
    dictionary = dictionary.merge(transforms, on="feature")
    dictionary["n_missing"] = frame[features].isna().sum().to_numpy()

    pairs = pd.read_parquet(features_root / "paired/raw-features.parquet", columns=PAIR_KEYS)
    store = pd.read_parquet(freeze / "transcript_store.parquet", columns=TRANSCRIPT_KEYS + ["transcript"])
    n_transcripts, n_pairs = len(frame), len(pairs)
    frame, pairs, n_answer_only = drop_answer_only(frame, pairs, store)
    print(f"answer-only transcripts dropped: {n_answer_only} "
          f"(transcript rows {n_transcripts}->{len(frame)}, pair rows {n_pairs}->{len(pairs)})")
    pairs = pd.concat([pairs, pair_differences(pairs, frame, features)], axis=1)     # matched on original keys
    frame, pairs = pool_aime(frame), pool_aime(pairs)
    accuracy = leave_item_out_accuracy(identity, split)
    frame = frame.merge(split, on=["bench", "item"], how="left", validate="many_to_one")
    frame = frame.merge(accuracy, on=["bench", "model", "item"], how="left", validate="many_to_one")
    pairs = pairs.merge(split, on=["bench", "item"], how="left", validate="many_to_one")
    for side in ("left", "right"):
        acc = accuracy.rename(columns={"model": f"model_{side}", "model_accuracy": f"accuracy_{side}"})
        pairs = pairs.merge(acc, on=["bench", "item", f"model_{side}"], how="left", validate="many_to_one")
    pairs["accuracy_difference"] = pairs.accuracy_left - pairs.accuracy_right
    pairs["left_stronger"] = np.sign(pairs.accuracy_difference).replace(0, np.nan)

    transcript, transcript_scaling = scale(frame, features, centre=True)
    pair, pair_scaling = scale(pairs, [f"d_{f}" for f in features], centre=False)
    return {"transcript.parquet": transcript, "pair.parquet": pair,
            "scaling-transcript.parquet": transcript_scaling, "scaling-pair.parquet": pair_scaling,
            "feature-dictionary.parquet": dictionary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/design_matrices.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    tables = build(cfg)
    for name, table in tables.items():
        if (out / name).exists():
            raise FileExistsError(f"refusing to overwrite {out / name}")
        table.to_parquet(out / name, index=False)
    d = tables["feature-dictionary.parquet"]
    print(f"transcripts={len(tables['transcript.parquet'])} pairs={len(tables['pair.parquet'])} "
          f"features={len(d)} -> {out}")
    print(d.groupby("block", sort=False).agg(n=("feature", "size"), log1p=("transform", lambda s: (s == "log1p").sum())).to_string())


if __name__ == "__main__":
    main()
