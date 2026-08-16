"""Build the eight truncation pair tables and collect the two rewrite tables.

The truncation rows use the exact campaign config and ``truncate_pair`` helper
used by the sampled-judge run.  Outputs are classifier-ready parquets for the
base/fine-tuned verdict-token evaluation.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge.dataset import truncate_pair
from prep_pairs import check_counterbalance, load_model_log, make_pair_rows

CONFIG_DIR = ROOT / "scripts/pairwise-newjudges/configs/manipulations"
MODELS_CSV = ROOT / "judge_results/judge_ft/models_ft.csv"
REWRITE_DIR = ROOT / "judge_results/rewrites"
DEFAULT_OUT = ROOT / "judge_results/judge_ft/manipulations/pairs"

TRUNCATIONS = (
    ("trunc_first_10", 0.10, "first"),
    ("trunc_first_50", 0.50, "first"),
    ("trunc_first_90", 0.90, "first"),
    ("trunc_first_100", 1.00, "first"),
    ("trunc_end_10", 0.10, "last"),
    ("trunc_end_50", 0.50, "last"),
    ("trunc_end_90", 0.90, "last"),
    ("trunc_end_100", 1.00, "last"),
)
REWRITES = ("paraphrase_haiku", "paraphrase_gpt")


def build_truncation_frame(cfg, metadata, caps, *, condition, frac, keep):
    rows = []
    for major, minor in cfg["comparisons"]:
        for item in cfg["sample_ids"][f"{major}-vs-{minor}"]:
            left, right = metadata[(major, item)], metadata[(minor, item)]
            text_left, text_right = truncate_pair(left["text"], right["text"], frac, keep)
            a = dict(model=major, cap=float(caps[major]), text=text_left,
                     options=left["options"])
            b = dict(model=minor, cap=float(caps[minor]), text=text_right,
                     options=right["options"])
            rows.extend(make_pair_rows(item, a, b, question=left["question"]))
    frame = pd.DataFrame(rows)
    frame["split"] = condition
    check_counterbalance(frame)
    return frame


def load_metadata(cfg, cache_dir):
    metadata = {}
    for model, relative_path in cfg["transcripts"].items():
        frame = load_model_log(str(ROOT / relative_path), cache_dir, cfg["max_length"])
        for row in frame.itertuples():
            metadata[(model, str(row.item))] = {
                "text": row.text,
                "options": row.options,
                "question": row.question,
            }
    return metadata


def write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def file_record(path):
    frame = pd.read_parquet(path, columns=["pair_uid", "order"])
    return {
        "file": path.name,
        "rows": len(frame),
        "pair_uids": int(frame.pair_uid.nunique()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load((CONFIG_DIR / "trunc_first_10.yaml").read_text())
    metadata = load_metadata(base_cfg, out / "cache")
    caps = pd.read_csv(MODELS_CSV).set_index("model").logit.to_dict()
    records = []

    for condition, frac, keep in TRUNCATIONS:
        cfg = yaml.safe_load((CONFIG_DIR / f"{condition}.yaml").read_text())
        frame = build_truncation_frame(
            cfg, metadata, caps, condition=condition, frac=frac, keep=keep)
        path = out / f"{condition}.parquet"
        write_frame(frame, path)
        records.append(file_record(path))
        print(f"{condition}: {len(frame)} rows, {frame.pair_uid.nunique()} pair-items")

    for condition in REWRITES:
        source = REWRITE_DIR / f"pairs_gpqa_{condition}.parquet"
        path = out / f"{condition}.parquet"
        if source.resolve() != path.resolve():
            shutil.copyfile(source, path)
        frame = pd.read_parquet(path)
        check_counterbalance(frame)
        records.append(file_record(path))
        print(f"{condition}: {len(frame)} rows, {frame.pair_uid.nunique()} pair-items")

    manifest = {"conditions": records, "n_conditions": len(records)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert len(records) == 10
    print(f"wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
