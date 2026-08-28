"""Write the two tables the verdict-mimic fine-tune needs, for staging to the cluster.

    uv run python scripts/verdict_dissection/mimic/prepare_mimic_data.py <data_dir>

<data_dir>/verdicts.parquet: one row per judge verdict draw (the judge GLM's rows) with the pair
keys, judge, pick_left and a role_<design> column (train / test / drop) per held-out design (model folds and strict folds).
<data_dir>/transcripts.parquet: the canonical-correct transcripts (AIME years pooled) with the
question and the rendered transcript the judge saw.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
from build_design_matrices import pool_aime          # noqa: E402
from prepare import PAIR_IDENTITY, designs, held_out  # noqa: E402

TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    verdicts = pd.read_parquet(Path(config["output_root"]) / "pair.parquet", columns=PAIR_IDENTITY + ["judge", "pick_left"])
    for design in designs():
        if design.startswith(("new-models-and-items", "model-fold")):
            verdicts[f"role_{design}"] = held_out(design, verdicts.bench, verdicts.item, verdicts.model_left, verdicts.model_right)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet",
                            columns=TRANSCRIPT_KEYS + ["sample_epoch", "question", "transcript", "correct", "is_canonical"])
    store = pool_aime(store[store.is_canonical & store.correct]).sort_values("sample_epoch").drop_duplicates(TRANSCRIPT_KEYS)
    transcripts = store[TRANSCRIPT_KEYS + ["question", "transcript"]].reset_index(drop=True)
    for side in ("left", "right"):
        keys = {k: f"{k}_{side}" if k in ("model", "epoch", "log") else k for k in TRANSCRIPT_KEYS}
        matched = verdicts.merge(transcripts[TRANSCRIPT_KEYS].rename(columns=keys), on=list(keys.values()), how="left", indicator=True)
        if (matched._merge != "both").any():
            raise ValueError(f"{int((matched._merge != 'both').sum())} verdicts have no {side} transcript")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    verdicts.to_parquet(args.data_dir / "verdicts.parquet", index=False)
    transcripts.to_parquet(args.data_dir / "transcripts.parquet", index=False)
    print(f"{len(verdicts)} verdicts, {len(transcripts)} transcripts")
    print(verdicts.filter(like="role_").apply(lambda c: c.value_counts()).to_string())


if __name__ == "__main__":
    main()
