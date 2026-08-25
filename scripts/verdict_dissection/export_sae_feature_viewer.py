"""Export the shared-SAE feature viewer data: per feature and benchmark, the description,
activation histogram and example transcripts drawn from the top, from activation quantiles,
from the most negative, and same-question zero-activation contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "item", "model", "epoch", "log"]
GROUPS = {"gpqa": ["gpqa"], "aime": ["aime2024", "aime2025", "aime2026"],
          "intercode": ["intercode"], "swe_bench_mini": ["swe_bench_mini"]}
N_FEATURES = 32
QUANTILE_BUCKETS = [("50-90%", 0.5, 0.9), ("10-50%", 0.1, 0.5), ("0-10%", 0.0, 0.1)]


def pick_rates(verdicts):
    """Mean judge pick rate per transcript, pooled over judges, orders and draws."""
    left = verdicts.rename(columns={"model_left": "model", "epoch_left": "epoch", "log_left": "log"})
    left = left[KEYS + ["pick_left"]].rename(columns={"pick_left": "picked"})
    right = verdicts.rename(columns={"model_right": "model", "epoch_right": "epoch", "log_right": "log"})
    right = right[KEYS + ["pick_left"]].assign(picked=lambda f: 1 - f.pick_left).drop(columns="pick_left")
    both = pd.concat([left, right]).astype({"picked": float})
    return both.groupby(KEYS).picked.agg(pick_rate="mean", pick_n="size").reset_index()


def candidates(candidates_json):
    return [{"description": c["description"], "correlation": c["correlation"], "p_value": c["p_value"]}
            for c in json.loads(candidates_json)]


def histogram(values, bins=40):
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.round(4).tolist()}


def by_model(rows, column):
    """Mean activation per model with the model's accuracy on this benchmark group."""
    table = rows.groupby("model").agg(mean_activation=(column, "mean"), accuracy=("accuracy", "mean"),
                                      n=(column, "size")).reset_index()
    table = table.sort_values("accuracy")
    return [{"model": r.model, "accuracy": None if pd.isna(r.accuracy) else float(r.accuracy),
             "mean_activation": float(r.mean_activation), "n": int(r.n)} for r in table.itertuples()]


def pick_by_bin(rows, column, n_bins=5):
    """Judges' mean pick rate for transcripts binned by activation: negative, zero, positive quantile bins."""
    judged = rows[rows.pick_n > 0]
    out = []
    for name, mask in [("negative", judged[column] < 0), ("zero", judged[column] == 0)]:
        part = judged[mask]
        out.append({"bin": name, "mean_pick": float(part.pick_rate.mean()) if len(part) else None,
                    "n": int(len(part)), "mean_activation": float(part[column].mean()) if len(part) else None})
    positive = judged[judged[column] > 0].sort_values(column)
    for k, index in enumerate(np.array_split(np.arange(len(positive)), n_bins)):
        part = positive.iloc[index]
        out.append({"bin": f"positive {k + 1}/{n_bins}", "mean_pick": float(part.pick_rate.mean()) if len(part) else None,
                    "n": int(len(part)), "mean_activation": float(part[column].mean()) if len(part) else None})
    return out


def example(row, activation, bucket):
    return {"activation": float(activation), "bucket": bucket, "model": row.model, "bench": row.bench,
            "item": row["item"], "epoch": int(row.epoch), "log": row.log,
            "accuracy": None if pd.isna(row.accuracy) else float(row.accuracy),
            "pick_rate": None if pd.isna(row.pick_rate) else float(row.pick_rate),
            "pick_n": 0 if pd.isna(row.pick_n) else int(row.pick_n),
            "split": row.split_role, "question": row.question, "transcript": row.transcript}


def choose_examples(rows, column, n_top, n_quantile, n_negative, rng):
    """Top, quantile-bucket, same-question contrast and most-negative rows, as (row, bucket)."""
    positive = rows[rows[column] > 0].sort_values(column, ascending=False)
    chosen = [(r, "top") for _, r in positive.head(n_top).iterrows()]
    rest = positive.iloc[n_top:]
    if len(rest):
        ranks = 1 - np.arange(len(rest)) / len(rest)  # 1 = highest remaining activation
        for name, low, high in QUANTILE_BUCKETS:
            bucket = rest[(ranks >= low) & (ranks < high)]
            if len(bucket):
                take = rng.choice(len(bucket), size=min(n_quantile, len(bucket)), replace=False)
                chosen += [(bucket.iloc[i], name) for i in sorted(take)]
    silent = rows[rows[column] == 0]
    for _, shown in positive.head(n_top).iterrows():
        same_item = silent[silent["item"] == shown["item"]]
        if len(same_item):
            chosen.append((same_item.iloc[rng.integers(len(same_item))], "contrast"))
    negative = rows[rows[column] < 0].sort_values(column)
    chosen += [(r, "negative") for _, r in negative.head(n_negative).iterrows()]
    return chosen


def feature_payload(feature, frame, descriptions, n_top, n_quantile, n_negative, seed):
    """One feature's viewer record: per group, description, histogram and bucketed examples."""
    column = f"sae_shared_{feature}"
    rng = np.random.default_rng(seed)
    groups = {}
    for group, benches in GROUPS.items():
        rows = frame[frame.bench.isin(benches)]
        values = rows[column].to_numpy()
        described = descriptions.get(f"shared:{feature}@{group}")
        groups[group] = {
            "n": int(len(rows)), "prevalence": float((values != 0).mean()) if len(rows) else 0.0,
            "histogram": histogram(values) if len(rows) else None,
            "by_model": by_model(rows, column) if len(rows) else [],
            "pick_by_bin": pick_by_bin(rows, column) if len(rows) else [],
            "description": None if described is None else {
                "text": described.description if isinstance(described.description, str) else None,
                "kept": bool(described.kept), "reason": described.reason if isinstance(described.reason, str) else None,
                "correlation": None if pd.isna(described.correlation) else float(described.correlation),
                "p_value": None if pd.isna(described.p_value) else float(described.p_value),
                "candidates": candidates(described.candidates_json) if isinstance(described.candidates_json, str) else [],
            },
            "examples": [example(r, r[column], bucket)
                         for r, bucket in choose_examples(rows, column, n_top, n_quantile, n_negative, rng)],
        }
    return {"feature": feature, "groups": groups}


def load_frame(tables_dir, freeze_dir, accuracy_path):
    columns = KEYS + ["sample_epoch", "content_hash", "split_role"] + [f"sae_shared_{i}" for i in range(N_FEATURES)]
    frame = pd.read_parquet(tables_dir / "transcript_features_qwen.parquet", columns=columns)
    store = pd.read_parquet(freeze_dir / "transcript_store.parquet",
                            columns=KEYS + ["sample_epoch", "content_hash", "question", "transcript"])
    frame = frame.merge(store, on=KEYS + ["sample_epoch", "content_hash"], how="left", validate="one_to_one")
    accuracy = pd.read_parquet(accuracy_path)[["model", "bench", "accuracy"]]
    frame = frame.merge(accuracy, on=["model", "bench"], how="left")
    verdicts = pd.read_parquet(freeze_dir / "verdicts.parquet",
                               columns=["bench", "item", "model_left", "model_right", "epoch_left",
                                        "epoch_right", "log_left", "log_right", "pick_left"])
    return frame.merge(pick_rates(verdicts), on=KEYS, how="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "judge_results/verdict_dissection/transcript_sae/tables-qwen")
    parser.add_argument("--freeze-dir", type=Path, default=ROOT / "judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66")
    parser.add_argument("--descriptions", type=Path, default=ROOT / "judge_results/verdict_dissection/transcript_sae/descriptions-qwen-shared/feature_descriptions.csv")
    parser.add_argument("--accuracy", type=Path, default=Path("/Volumes/T7/judge-dissection/transcript-features/raw/model-benchmark-accuracy.parquet"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "interactive_dashboards/sae_features/data")
    parser.add_argument("--n-top", type=int, default=10)
    parser.add_argument("--n-quantile", type=int, default=4, help="examples per quantile bucket")
    parser.add_argument("--n-negative", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = load_frame(args.tables_dir, args.freeze_dir, args.accuracy)
    descriptions = {r.feature: r for r in pd.read_csv(args.descriptions).itertuples()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for feature in range(N_FEATURES):
        payload = feature_payload(feature, frame, descriptions, args.n_top, args.n_quantile, args.n_negative,
                                  args.seed + feature)
        (args.output_dir / f"feature_{feature:02d}.json").write_text(json.dumps(payload))
        index.append({"feature": feature, "groups": {
            g: {"prevalence": v["prevalence"], "kept": bool(v["description"] and v["description"]["kept"]),
                "text": v["description"] and v["description"]["text"]} for g, v in payload["groups"].items()}})
        print(f"feature {feature}: " + ", ".join(f"{g} {len(v['examples'])}" for g, v in payload["groups"].items()))
    (args.output_dir / "index.json").write_text(json.dumps({"features": index, "groups": list(GROUPS)}))


if __name__ == "__main__":
    main()
