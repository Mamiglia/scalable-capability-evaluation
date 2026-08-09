"""Stage recording inputs onto the shared storage.

Builds staging/inputs.parquet: one row per elo entity (text_id, item)
with the question, the multiple-choice options as the author model saw
them, and the transcript text. Also creates the storage dirs the
cluster job expects. Existing files are never touched: if the target
parquet exists the script stops and says so.

Runs locally (needs inspect_ai and the eval logs); the cluster job then
needs nothing beyond inputs.parquet and the model weights.

Usage:
  uv run python scripts/analyses/judge_interp/stage_hpc.py
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

sys.path += [str(Path(__file__).resolve().parents[2]),
             str(Path(__file__).resolve().parents[1])]
from analyses.digest_verdicts import pair_items

ROOT = Path(__file__).resolve().parents[3]

OPTIONS_BLOCK = """Multiple choice options as shown to the model:
{options}

"""


def text_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def options_block(sample) -> str:
    """The sample's options in the order that model saw them."""
    choices = getattr(sample, "choices", None)
    if not choices:
        return ""
    return OPTIONS_BLOCK.format(options="\n".join(
        f"{chr(ord('A') + i)}) {c.strip()}" for i, c in enumerate(choices)))


def storage_root(cfg) -> Path:
    roots = [Path(r) for r in cfg["storage_roots"]]
    for root in roots:
        if root.is_dir():
            return root
    raise SystemExit(f"no storage root mounted; tried {roots}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    storage = storage_root(cfg)
    target = storage / cfg["record"]["inputs"]
    if target.exists():
        raise SystemExit(f"{target} already exists; not touching it")

    rows = pd.read_parquet(ROOT / cfg["rows_parquet"])
    elo = pd.read_csv(ROOT / cfg["out_dir"] / "elo.csv")
    texts = pd.read_parquet(ROOT / cfg["texts_parquet"]).set_index("text_id")["text"]

    # (text_id, item) -> question + options, via the same rebuild the
    # digester verified against the logged judge inputs.
    entries = {}
    conflicts = 0
    for log_file in sorted(rows["log_file"].unique()):
        meta = read_eval_log(str(ROOT / log_file), header_only=True).eval.metadata
        assert not meta["include_reasoning"], f"{log_file} judged with reasoning"
        items = pair_items(ROOT / meta["major_path"], ROOT / meta["minor_path"],
                           include_reasoning=False,
                           require_both_correct=meta["require_both_correct"])
        for item, it in items.items():
            for text, sample in ((it.t_major, it.s_major), (it.t_minor, it.s_minor)):
                key = (text_id(text), item)
                entry = dict(question=it.question, options=options_block(sample))
                if key in entries:
                    conflicts += entries[key] != entry
                else:
                    entries[key] = entry

    inputs = pd.DataFrame([
        dict(text_id=tid, item=item, model=e_row["model"],
             question=entries[(tid, item)]["question"],
             options=entries[(tid, item)]["options"],
             text=texts[tid])
        for tid, item, e_row in
        ((r["text_id"], r["item"], r) for _, r in elo.iterrows())
    ])
    assert len(inputs) == len(elo), "inputs must cover every elo entity"
    print(f"{len(inputs)} entities, {conflicts} option/question conflicts (expect 0)")

    for d in ("staging", "logs", cfg["record"]["acts_dir"], "models"):
        (storage / d).mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(target, index=False)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
