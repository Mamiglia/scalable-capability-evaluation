"""Paired full-transcript vs truncated or channel-filtered direct forecasts, per judge, semester and condition.

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
from challenges.truncation.run import CHANNELS_ROOT, ROOT as TRUNCATION_ROOT

REPO = Path(__file__).resolve().parents[3]
KEY = ["model", "bench", "item"]


def baseline_ratings(domain, judge, semester):
    """The full-transcript campaign's filtered rating export, the same file prepare.py froze against."""
    source = REPO / "artifacts" / CAMPAIGNS[domain][0]
    return source / ("consolidated/results" if len(CAMPAIGNS[domain]) > 1 else "results") / judge / semester / "ratings.csv"


def forecast(ratings, refs, truth):
    prediction = calibrate(ratings, refs).reindex(truth.index)
    return prediction, score(prediction, truth)


def truncated_ratings(payload, condition, ROOT):
    domain, semester, judge = payload["domain"], payload["semester"], payload["judge"]
    channels = payload.get("channels")
    published = f"absolute/channels-{channels}" if channels else f"absolute/truncated-{payload['truncate']}-{payload['max_length']}"
    identity = lambda c: (c.get("max_length"), c.get("truncate"), c.get("channels"))
    for base in (REPO / "logs", ROOT / "publication"):
        for path in sorted((base / published / domain / semester / judge).glob("*/condition.json")):
            condition_json = json.loads(path.read_text())
            if identity(condition_json) == identity(payload):
                return pd.read_csv(path.parent / "ratings.csv", dtype={"item": str}), str(path.parent.relative_to(base))
    local = ROOT / "results" / condition / "ratings.csv"
    if local.exists():
        return pd.read_csv(local, dtype={"item": str}), "local results (unaudited)"
    return None, None


def main(ROOT, name):
    rows = []
    for summary in json.loads((ROOT / "sweep-summary.json").read_text()):
        if summary["status"] != "prepared":
            continue
        cut = summary.get("condition") or f"{summary['truncate']}-{summary['max_length']}"
        condition = f"{summary['domain']}/{summary['judge']}/{summary['semester']}/{cut}"
        payload = json.loads((ROOT / "inputs" / condition / "job.json").read_text())
        truncated, origin = truncated_ratings(payload, condition, ROOT)
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
                     "condition": cut, "truncate": summary["truncate"], "max_length": summary["max_length"], "source": origin,
                     "excluded_inputs": summary.get("excluded_inputs", 0),
                     "retained_inputs": retained, "truncated_inputs": summary["changed_inputs"],
                     "truncated_reference_inputs": summary["truncated_reference_inputs"],
                     "truncated_target_inputs": summary["truncated_target_inputs"],
                     "full_ratings": len(full), "truncated_ratings": len(truncated), "matched_ratings": len(shared),
                     "full_coverage": len(full) / (retained + summary.get("excluded_inputs", 0)), "truncated_coverage": len(truncated) / retained,
                     "full_forecasts": int(pf.notna().sum()), "truncated_forecasts": int(pt.notna().sum()),
                     "full_mse": sf["mse"], "truncated_mse": st["mse"],
                     "matched_full_mse": sfm["mse"], "matched_truncated_mse": stm["mse"],
                     "mean_forecast_shift_truncated_minus_full": float((ptm - pfm).mean()),
                     "mean_target_rating_full": float(full_m[full_m.model.isin(truth.index)].score.mean()),
                     "mean_target_rating_truncated": float(trunc_m[trunc_m.model.isin(truth.index)].score.mean())})
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "comparison"
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["domain", "semester", "judge", "condition"])
    if name == "channels":  # a channel condition filters the transcript rather than truncating it
        frame = frame.drop(columns=["truncate", "max_length"]).rename(columns=lambda c: c.replace("truncated", "condition"))
    frame.to_csv(out / f"{name}-comparison.csv", index=False)
    cells = frame.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    (out / f"{name}-comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    for root, name in ((TRUNCATION_ROOT, "truncation"), (CHANNELS_ROOT, "channels")):
        if (root / "sweep-summary.json").exists():
            main(root, name)
