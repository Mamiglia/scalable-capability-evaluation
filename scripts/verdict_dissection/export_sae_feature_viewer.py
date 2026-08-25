"""Export the shared-SAE feature viewer data: per feature, descriptions, histograms and example transcripts."""
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


def example(row, activation):
    return {"activation": float(activation), "model": row.model, "bench": row.bench, "item": row["item"],
            "epoch": int(row.epoch), "accuracy": None if pd.isna(row.accuracy) else float(row.accuracy),
            "pick_rate": None if pd.isna(row.pick_rate) else float(row.pick_rate),
            "pick_n": 0 if pd.isna(row.pick_n) else int(row.pick_n),
            "split": row.split_role, "question": row.question, "transcript": row.transcript}


def feature_payload(feature, frame, descriptions, n_examples, seed):
    """One feature's viewer record: per group, description, histogram, firing / negative / contrast examples."""
    column = f"sae_shared_{feature}"
    rng = np.random.default_rng(seed)
    groups = {}
    for group, benches in GROUPS.items():
        rows = frame[frame.bench.isin(benches)]
        values = rows[column].to_numpy()
        firing = rows.iloc[np.argsort(-values)[:n_examples]]
        firing = firing[firing[column] > 0]
        negative = rows.iloc[np.argsort(values)[:n_examples]]
        negative = negative[negative[column] < 0]
        silent = rows[rows[column] == 0]
        contrasts = []
        for _, shown in firing.iterrows():
            same_item = silent[silent["item"] == shown["item"]]
            if len(same_item):
                contrasts.append(same_item.iloc[rng.integers(len(same_item))])
        described = descriptions.get(f"shared:{feature}@{group}")
        groups[group] = {
            "n": int(len(rows)), "prevalence": float((values != 0).mean()) if len(rows) else 0.0,
            "histogram": histogram(values) if len(rows) else None,
            "description": None if described is None else {
                "text": described.description if isinstance(described.description, str) else None,
                "kept": bool(described.kept), "reason": described.reason if isinstance(described.reason, str) else None,
                "correlation": None if pd.isna(described.correlation) else float(described.correlation),
                "p_value": None if pd.isna(described.p_value) else float(described.p_value),
                "candidates": candidates(described.candidates_json) if isinstance(described.candidates_json, str) else [],
            },
            "firing": [example(r, r[column]) for _, r in firing.iterrows()],
            "negative": [example(r, r[column]) for _, r in negative.iterrows()],
            "contrast": [example(r, 0.0) for r in contrasts],
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
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = load_frame(args.tables_dir, args.freeze_dir, args.accuracy)
    descriptions = {r.feature: r for r in pd.read_csv(args.descriptions).itertuples()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for feature in range(N_FEATURES):
        payload = feature_payload(feature, frame, descriptions, args.n_examples, args.seed + feature)
        (args.output_dir / f"feature_{feature:02d}.json").write_text(json.dumps(payload))
        index.append({"feature": feature, "groups": {
            g: {"prevalence": v["prevalence"], "kept": bool(v["description"] and v["description"]["kept"]),
                "text": v["description"] and v["description"]["text"]} for g, v in payload["groups"].items()}})
        print(f"feature {feature}: " + ", ".join(
            f"{g} {len(v['firing'])}+{len(v['negative'])}-/{len(v['contrast'])}c" for g, v in payload["groups"].items()))
    (args.output_dir / "index.json").write_text(json.dumps({"features": index, "groups": list(GROUPS)}))


if __name__ == "__main__":
    main()
