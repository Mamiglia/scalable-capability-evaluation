"""Extract as-judged transcripts and verdict rows for the score model (untruncated runs only).

texts.parquet holds one row per unique transcript 
rows.parquet one row per usable verdict

Usage:
  uv run python scripts/analyses/score_prediction/extract_texts.py
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analyses.digest_verdicts import pair_items

# Repo root; config and log paths are root-relative, so anchor them here
# instead of assuming the caller's cwd (PyCharm runs from the script dir).
ROOT = Path(__file__).resolve().parents[3]


def text_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    verdicts = pd.read_csv(ROOT / cfg["verdicts_csv"])
    n_total = len(verdicts)
    verdicts = verdicts[verdicts["truncate_frac"].isna()]
    n_full = len(verdicts)
    verdicts = verdicts[verdicts["picked"].notna()]
    n_parseable = len(verdicts)
    # NaN length means the digester's rebuild did not match the logged judge input
    verdicts = verdicts[verdicts["len_major"].notna() & verdicts["len_minor"].notna()]
    n_kept = len(verdicts)

    texts: dict[str, dict] = {}
    rows = []
    n_len_mismatch = n_conflict = n_identical = 0
    
    for log_file, log_verdicts in verdicts.groupby("log_file"):
        meta = read_eval_log(str(ROOT / log_file), header_only=True).eval.metadata
        # Reasoning tokens are never included; refuse runs judged with them.
        assert not meta["include_reasoning"], f"{log_file} judged with reasoning"
        items = pair_items(ROOT / meta["major_path"], ROOT / meta["minor_path"],
                           include_reasoning=False,
                           require_both_correct=meta["require_both_correct"])
        for verdict in log_verdicts.itertuples():
            pair_item = items[str(verdict.item)]
            for text, sample, model, path, reasoning_len, expected_len in (
                (pair_item.t_major, pair_item.s_major, verdict.major,
                 meta["major_path"], pair_item.reasoning_len_major, verdict.len_major),
                (pair_item.t_minor, pair_item.s_minor, verdict.minor,
                 meta["minor_path"], pair_item.reasoning_len_minor, verdict.len_minor),
            ):
                if len(text) != int(expected_len):
                    n_len_mismatch += 1
                tid = text_id(text)
                existing = texts.get(tid)
                if existing is None:
                    texts[tid] = dict(
                        text_id=tid, model=model, item=str(verdict.item),
                        epoch=getattr(sample, "epoch", 0), text=text,
                        n_chars=len(text), reasoning_len=reasoning_len,
                        source_log=str(path))
                elif existing["model"] != model:
                    n_conflict += 1

            tid_major = text_id(pair_item.t_major)
            tid_minor = text_id(pair_item.t_minor)
            n_identical += tid_major == tid_minor
            slot1, slot2 = ((tid_major, tid_minor) if str(verdict.slot_of_major) == "1"
                            else (tid_minor, tid_major))
            rows.append(dict(
                row_id=(f"{verdict.judge}|{verdict.major}|{verdict.minor}"
                        f"|{verdict.item}|{verdict.slot_of_major}"),
                judge=verdict.judge, major=verdict.major, minor=verdict.minor,
                item=str(verdict.item), slot_of_major=str(verdict.slot_of_major),
                y=int(int(verdict.picked) == 1), confidence=verdict.confidence,
                text_id_slot1=slot1, text_id_slot2=slot2,
                pair_key="|".join(sorted([verdict.major, verdict.minor])),
                phase=verdict.phase, log_file=str(log_file)))

    texts_df = pd.DataFrame(texts.values())
    rows_df = pd.DataFrame(rows)
    assert rows_df["row_id"].is_unique, "duplicate row_ids"
    texts_df.to_parquet(out / "texts.parquet", index=False)
    rows_df.to_parquet(out / "rows.parquet", index=False)

    print(f"{n_total} verdict rows: {n_total - n_full} truncated excluded, "
          f"{n_full - n_parseable} unparseable dropped, "
          f"{n_parseable - n_kept} len-NaN (rebuild mismatch) dropped, {n_kept} kept")
    print(f"length check: {n_len_mismatch} mismatches (expect 0)")
    print(f"{len(texts_df)} unique texts, {n_conflict} cross-model hash conflicts "
          f"(expect 0), {n_identical} rows with identical slot texts")
    print(f"\ntexts per model:")
    print(texts_df.groupby("model").agg(
        n=("text_id", "size"), mean_chars=("n_chars", "mean")).round(0)
        .to_string())
    print(f"\nrows per judge:")
    print(rows_df.groupby("judge").agg(
        n=("row_id", "size"), y_rate=("y", "mean")).round(3).to_string())
    print(f"\n-> {out}/texts.parquet, {out}/rows.parquet")


if __name__ == "__main__":
    main()