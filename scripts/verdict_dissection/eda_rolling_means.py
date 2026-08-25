"""Capability correlations for raw and position-adjusted rolling means."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from eda_capability_correlations import capability_correlations, summarise

KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]
METRICS = ("short", "long", "long_question", "context_gain", "question_gain")


def position_bins(token_index):
    """Bins 0–31, 32–63, 64–127, then successive doublings."""
    token_index = np.asarray(token_index, dtype=np.int64)
    bins = np.zeros(len(token_index), dtype=np.int16)
    later = token_index >= 32
    bins[later] = 1 + np.floor(np.log2(token_index[later] / 32)).astype(np.int16)
    return bins


def _batches(path, columns, batch_size):
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    return dataset.scanner(columns=columns, filter=ds.field("assistant") == True,
                           batch_size=batch_size).to_batches()  # noqa: E712


def _position_baselines(path, batch_size):
    parts = []
    columns = ["scorer", "bench", "token_index", "assistant"] + [
        f"{metric}_bits" for metric in METRICS]
    for batch in _batches(path, columns, batch_size):
        frame = batch.to_pandas()
        frame["position_bin"] = position_bins(frame.token_index)
        for metric in METRICS:
            values = frame[f"{metric}_bits"]
            valid = values.notna()
            if not valid.any():
                continue
            work = frame.loc[valid, ["scorer", "bench", "position_bin"]].copy()
            work["bits"] = values[valid].to_numpy()
            work["squares"] = work.bits ** 2
            grouped = (work.groupby(["scorer", "bench", "position_bin"],
                                    as_index=False, observed=True)
                       .agg(n_tokens=("bits", "size"), sum_bits=("bits", "sum"),
                            sum_squares=("squares", "sum")))
            grouped["metric"] = metric
            parts.append(grouped)
    totals = (pd.concat(parts, ignore_index=True)
              .groupby(["scorer", "bench", "position_bin", "metric"],
                       as_index=False, observed=True)[
                  ["n_tokens", "sum_bits", "sum_squares"]].sum())
    totals["mean_bits"] = totals.sum_bits / totals.n_tokens
    variance = totals.sum_squares / totals.n_tokens - totals.mean_bits ** 2
    totals["sd_bits"] = np.sqrt(variance.clip(lower=0))
    bins = totals.position_bin.to_numpy(dtype=np.int64)
    starts = np.zeros(len(totals), dtype=np.int64)
    starts[bins > 0] = 32 * 2 ** (bins[bins > 0] - 1)
    totals["position_start"] = starts
    totals["position_end_exclusive"] = np.where(bins == 0, 32, 2 * starts)
    return totals[["scorer", "bench", "metric", "position_bin", "position_start",
                   "position_end_exclusive", "n_tokens", "mean_bits", "sd_bits"]]


def _adjusted_transcript_means(path, baselines, batch_size,
                               normalised_tokens=None):
    columns = KEYS + ["scorer", "token_index", "assistant"] + [
        f"{metric}_bits" for metric in METRICS]
    lookups = {
        metric: (baselines[baselines.metric == metric]
                 .set_index(["scorer", "bench", "position_bin"]).mean_bits)
        for metric in METRICS
    }
    parts, value_columns, token_rows = [], [], 0
    for metric in METRICS:
        value_columns.extend([f"{metric}_sum", f"{metric}_count"])
    for part, batch in enumerate(_batches(path, columns, batch_size)):
        frame = batch.to_pandas()
        frame["position_bin"] = position_bins(frame.token_index)
        index = pd.MultiIndex.from_arrays(
            [frame.scorer, frame.bench, frame.position_bin],
            names=["scorer", "bench", "position_bin"])
        aggregations = {}
        for metric in METRICS:
            values = frame[f"{metric}_bits"].to_numpy(dtype=float)
            expected = lookups[metric].reindex(index).to_numpy()
            missing = np.isfinite(values) & ~np.isfinite(expected)
            if missing.any():
                raise ValueError(f"missing positional baseline for {metric}")
            residual = f"_{metric}_residual"
            frame[residual] = values - expected
            aggregations[f"{metric}_sum"] = (residual, "sum")
            aggregations[f"{metric}_count"] = (residual, "count")
        if normalised_tokens is not None:
            token_frame = frame[KEYS + ["scorer", "token_index", "position_bin"]].copy()
            for metric in METRICS:
                token_frame[f"{metric}_position_adjusted_bits"] = frame[
                    f"_{metric}_residual"]
            pq.write_to_dataset(
                pa.Table.from_pandas(token_frame, preserve_index=False),
                root_path=normalised_tokens, partition_cols=["scorer", "bench"],
                basename_template=f"part-{part}-{{i}}.parquet")
            token_rows += len(token_frame)
            if part % 20 == 0:
                print(f"normalised token rows {token_rows}", flush=True)
        parts.append(frame.groupby(KEYS + ["scorer"], as_index=False,
                                   observed=True).agg(**aggregations))
    totals = (pd.concat(parts, ignore_index=True)
              .groupby(KEYS + ["scorer"], as_index=False, observed=True)[
                  value_columns].sum())
    mean_columns = []
    for metric in METRICS:
        column = f"_{metric}_mean"
        totals[column] = totals[f"{metric}_sum"] / totals[f"{metric}_count"].replace(0, np.nan)
        mean_columns.append(column)
    wide = totals.set_index(KEYS + ["scorer"])[mean_columns].unstack("scorer")
    wide.columns = [
        f"surprisal_rolling_{metric.removeprefix('_').removesuffix('_mean')}_"
        f"position_adjusted_mean__{scorer}"
        for metric, scorer in wide.columns]
    return wide.reset_index()


def compute_position_adjustment(token_dataset, batch_size=500_000,
                                normalised_tokens=None):
    baselines = _position_baselines(token_dataset, batch_size)
    adjusted = _adjusted_transcript_means(
        token_dataset, baselines, batch_size, normalised_tokens)
    return baselines, adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token_dataset", type=Path)
    parser.add_argument("transcript_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--min-models", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    stage = args.output_dir.with_name(f".{args.output_dir.name}.staging")
    if stage.exists():
        raise FileExistsError(f"staging directory already exists: {stage}")
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    try:
        baselines, adjusted = compute_position_adjustment(
            args.token_dataset, args.batch_size,
            normalised_tokens=stage / "token-predictability")
        raw_features = [
            f"surprisal_rolling_{metric}_mean__{scorer}"
            for metric in METRICS for scorer in ("llama", "qwen", "gemma")]
        adjusted_features = [column for column in adjusted if "position_adjusted" in column]
        transcripts = pd.read_parquet(
            args.transcript_features, columns=KEYS + ["accuracy"] + raw_features)
        transcripts = transcripts.merge(adjusted, on=KEYS, validate="one_to_one")
        dictionary = pd.DataFrame({
            "feature": raw_features + adjusted_features,
            "group": (["rolling-mean"] * len(raw_features) +
                      ["rolling-position-adjusted-mean"] * len(adjusted_features)),
        })
        by_item = capability_correlations(transcripts, dictionary, args.min_models)
        overall, by_benchmark = summarise(by_item)
        baselines.to_parquet(stage / "position-baselines.parquet", index=False)
        adjusted.to_parquet(stage / "transcript-means.parquet", index=False)
        eda = stage / "capability-eda"
        eda.mkdir()
        by_item.to_parquet(eda / "by-item.parquet", index=False)
        by_benchmark.to_parquet(eda / "by-benchmark.parquet", index=False)
        overall.to_parquet(eda / "summary.parquet", index=False)
        overall.to_csv(eda / "summary.csv", index=False)
        pd.DataFrame([{
            "metrics": ",".join(METRICS), "summary": "mean assistant-token bits",
            "position_bins": "0-31, 32-63, 64-127, then doubling",
            "baseline_scope": "benchmark x scorer x metric x position_bin",
            "adjustment": "token bits minus positional mean",
            "token_rows": "assistant tokens only; raw tokens remain unchanged",
            "min_models": args.min_models,
            "model_fit_rule": "re-estimate positional baselines within training folds",
        }]).to_parquet(stage / "settings.parquet", index=False)
        os.rename(stage, args.output_dir)
        print(f"baselines={len(baselines)} transcripts={len(adjusted)} "
              f"correlations={len(by_item)}", flush=True)
    except Exception:
        print(f"failed output remains at {stage}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
