"""Derive data/meta/coding_frontier/ from data/meta/coding/: the frontier models as extra 2026-I targets.

    PYTHONPATH=src .venv/bin/python src/challenges/frontier/integrate.py [--write]

The frozen coding tables stay untouched (a refreeze would reshuffle the train set). Here:
- models.csv  = coding's rows, the original 2026-I targets' role blanked, plus one row per frontier
                model: paths to its subset logs, `2026-I_role = target`, theta* from
                `reference.freeze.local_fit` on the 2026-I HARD items it answered (a, b, guess pinned
                to the global fit, so theta* lands on the same scale as everyone else's).
- items.csv   = coding's (same train set, same HARD set).
- responses.csv = coding's plus the frontier models' rows; unscored samples (refusals, errors,
                cancelled) are dropped, not counted wrong (`evals.responses` would count them wrong).
- fit.yml     = coding's plus provenance.

Then `uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I ...` scores protocols on the
frontier targets alone, against the same references and items as the coding 2026-I task.
"""

import argparse
import datetime
import glob
from pathlib import Path

import pandas as pd
import yaml

from configs.paths import is_successful_eval
from evals.sample import extract_score
from inspect_ai.log import read_eval_log_sample_summaries
from reference.freeze import local_fit

SOURCE = Path("data/meta/coding")
OUT = Path("data/meta/coding_frontier")
SEMESTER = "2026-I"
SUBSETS = {"terminal_bench_2_1": "0af2ba2c", "swe_bench_mini": "2d1e487f"}


def frontier_logs() -> dict[str, dict[str, Path]]:
    """model -> bench -> its repeat-0 log under the subset hashes."""
    found: dict[str, dict[str, Path]] = {}
    for bench, h in SUBSETS.items():
        for p in sorted(glob.glob(f"data/{bench}/{h}/*/0/*.eval")):
            path = Path(p)
            if is_successful_eval(path):
                found.setdefault(path.parts[3], {})[bench] = path
    return found


def responses_of(model: str, logs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for bench, path in logs.items():
        for s in read_eval_log_sample_summaries(str(path)):
            if not s.scores:  # refusal / error / cancelled: not scored, not wrong
                continue
            rows.append(dict(bench=bench, item=str(s.id), model=model, repeat=0, correct=bool(extract_score(s))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()

    models = pd.read_csv(SOURCE / "models.csv", index_col="model")
    items = pd.read_csv(SOURCE / "items.csv", dtype={"item": str})
    responses = pd.read_csv(SOURCE / "responses.csv", dtype={"item": str})
    settings = yaml.safe_load((SOURCE / "fit.yml").read_text())

    logs = frontier_logs()
    new = pd.concat([responses_of(m, l) for m, l in logs.items()], ignore_index=True)
    hard = items[items[f"{SEMESTER}_set"] == "hard"][["bench", "item", "a", "b", "guess"]]
    n_hard = new.merge(hard[["bench", "item"]]).groupby("model").size()
    print("frontier models:", {m: sorted(l) for m, l in logs.items()})
    print("scored responses per model:", new.groupby("model").size().to_dict(), "| on HARD:", n_hard.to_dict())

    theta = local_fit(new, hard, sorted(logs), warmup=args.warmup, samples=args.samples, seed=settings.get("seed", 0))
    refs = models[models[f"{SEMESTER}_role"] == "reference"][f"{SEMESTER}_theta"]
    print(f"\n{SEMESTER} theta* (references span {refs.min():.2f} .. {refs.max():.2f}):")
    print(theta.round(3).to_string())

    out_models = models.copy()
    out_models.loc[out_models[f"{SEMESTER}_role"] == "target", f"{SEMESTER}_role"] = ""
    for m, l in logs.items():
        row = {c: "" if c.endswith("_role") else float("nan") for c in models.columns}
        row.update({f"path_{b}": str(p.relative_to("data")) for b, p in l.items()})
        row.update({"semester": "", f"{SEMESTER}_role": "target",
                    f"{SEMESTER}_theta": theta.loc[m, "theta"], f"{SEMESTER}_theta_sd": theta.loc[m, "theta_sd"]})
        out_models.loc[m] = pd.Series(row)
    out_responses = pd.concat([responses, new[["bench", "item", "model", "repeat", "correct"]]], ignore_index=True)
    settings = {**settings, "domain": "coding_frontier", "derived_from": "coding",
                "derived_at": datetime.date.today().isoformat(),
                "frontier_targets": sorted(logs), "frontier_hashes": SUBSETS,
                "note": "coding's frozen tables + frontier models as extra 2026-I targets (theta* = local_fit on "
                        "their scored 2026-I HARD responses); original 2026-I targets' role blanked; unscored "
                        "(refused) samples excluded from responses.csv"}
    if not args.write:
        print("\ndry run; pass --write to create", OUT)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    out_models.to_csv(OUT / "models.csv", float_format="%.4f")
    items.to_csv(OUT / "items.csv", index=False)
    out_responses.sort_values(["bench", "item", "model", "repeat"]).to_csv(OUT / "responses.csv", index=False)
    (OUT / "fit.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    print("wrote", OUT, ":", len(out_models), "models,", len(items), "items,", len(out_responses), "responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
