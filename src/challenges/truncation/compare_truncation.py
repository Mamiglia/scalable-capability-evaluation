"""Paired full-transcript vs truncated direct forecasts, per judge, semester, cut and limit.

Full-transcript ratings are the visible campaign's filtered export (retained-set mask applied);
truncated ratings come from the published condition, its publication stage, or the local results.
Both are recalibrated on the same previous-semester references and scored against the same theta*.
"""
import json
from pathlib import Path

import pandas as pd

from capability.evaluate import score
from protocols.direct_judge.scoring import calibrate
from challenges.truncation.prepare import CAMPAIGNS

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/direct-judging-truncation"  # frozen inputs, logs, results
KEY = ["model", "bench", "item"]


def baseline_ratings(domain, judge, semester):
    """The full-transcript campaign's filtered rating export, the same file prepare.py froze against."""
    source = REPO / "artifacts" / CAMPAIGNS[domain][0]
    return source / ("consolidated/results" if len(CAMPAIGNS[domain]) > 1 else "results") / judge / semester / "ratings.csv"


def forecast(ratings, refs, truth):
    prediction = calibrate(ratings, refs).reindex(truth.index)
    return prediction, score(prediction, truth)


def truncated_ratings(payload, condition):
    domain, semester, judge = payload["domain"], payload["semester"], payload["judge"]
    hashes = {p.parent.name for p in (ROOT / "results" / condition / "logs").glob("*")} if (ROOT / "results" / condition / "logs").exists() else set()
    for base in (REPO / "logs", ROOT / "publication"):
        for path in sorted((base / f"direct_truncated-{payload['truncate']}-{payload['max_length']}" / domain / semester / judge).glob("*/condition.json")):
            condition_json = json.loads(path.read_text())
            if (condition_json.get("max_length"), condition_json.get("truncate")) == (payload["max_length"], payload["truncate"]):
                return pd.read_csv(path.parent / "ratings.csv", dtype={"item": str}), str(path.parent.relative_to(base))
    local = ROOT / "results" / condition / "ratings.csv"
    if local.exists():
        return pd.read_csv(local, dtype={"item": str}), "local results (unaudited)"
    return None, None


def main():
    rows = []
    for summary in json.loads((ROOT / "sweep-summary.json").read_text()):
        if summary["status"] != "prepared":
            continue
        condition = f"{summary['domain']}/{summary['judge']}/{summary['semester']}/{summary['truncate']}-{summary['max_length']}"
        payload = json.loads((ROOT / "inputs" / condition / "job.json").read_text())
        truncated, origin = truncated_ratings(payload, condition)
        if truncated is None:
            print(f"{condition}: no truncated ratings yet"); continue
        full = pd.read_csv(baseline_ratings(summary["domain"], summary["judge"], summary["semester"]), dtype={"item": str})
        full["item"], truncated["item"] = full["item"].astype(str), truncated["item"].astype(str)
        refs, truth = pd.Series(payload["references"]), pd.Series(payload["truth"])
        shared = full[KEY].merge(truncated[KEY])
        full_m, trunc_m = full.merge(shared), truncated.merge(shared)
        pf, sf = forecast(full, refs, truth)
        pt, st = forecast(truncated, refs, truth)
        pfm, sfm = forecast(full_m, refs, truth)
        ptm, stm = forecast(trunc_m, refs, truth)
        retained = summary["total_ratings"]
        rows.append({"domain": summary["domain"], "semester": summary["semester"], "judge": summary["judge"],
                     "truncate": summary["truncate"], "max_length": summary["max_length"], "source": origin,
                     "retained_inputs": retained, "truncated_inputs": summary["changed_inputs"],
                     "truncated_reference_inputs": summary["truncated_reference_inputs"],
                     "truncated_target_inputs": summary["truncated_target_inputs"],
                     "full_ratings": len(full), "truncated_ratings": len(truncated), "matched_ratings": len(shared),
                     "full_coverage": len(full) / retained, "truncated_coverage": len(truncated) / retained,
                     "full_forecasts": int(pf.notna().sum()), "truncated_forecasts": int(pt.notna().sum()),
                     "full_mse": sf["mse"], "truncated_mse": st["mse"],
                     "matched_full_mse": sfm["mse"], "matched_truncated_mse": stm["mse"],
                     "mean_forecast_shift_truncated_minus_full": float((ptm - pfm).mean()),
                     "mean_target_rating_full": float(full_m[full_m.model.isin(truth.index)].score.mean()),
                     "mean_target_rating_truncated": float(trunc_m[trunc_m.model.isin(truth.index)].score.mean())})
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "comparison"
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["domain", "semester", "judge", "truncate", "max_length"])
    frame.to_csv(out / "truncation-comparison.csv", index=False)
    cells = frame.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    (out / "truncation-comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
