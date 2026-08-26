"""Judge logs -> verdict table for ranking.rank: bench, judge, item, model_left, model_right, pick_left."""

from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries


def from_logs(log_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(log_dir).glob("*.eval")):
        log = read_eval_log(str(path), header_only=True)
        if log.status != "success":
            continue
        md = log.eval.metadata or {}
        major, minor = md.get("major") or md["name_major"], md.get("minor") or md["name_minor"]
        bench = md.get("bench") or md["benchmark_name"]
        for s in read_eval_log_sample_summaries(str(path)):
            answer = next(iter((s.scores or {}).values())).answer
            if answer not in ("1", "2"):
                continue
            item, target = str(s.id).rsplit("_", 1)
            left, right = (major, minor) if target == "1" else (minor, major)
            rows.append(dict(bench=bench, judge=log.eval.model, item=item, model_left=left,
                             model_right=right, pick_left=float(answer == "1")))
    return pd.DataFrame(rows, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
