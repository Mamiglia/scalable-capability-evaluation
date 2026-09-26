"""Per-condition progress: rated samples in successful or in-flight logs versus frozen new ratings."""
import json
from pathlib import Path
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

from challenges.truncation.run import CHANNELS_ROOT, ROOT

for summary in [s for r in (ROOT, CHANNELS_ROOT) if (r / "sweep-summary.json").exists()
                for s in json.loads((r / "sweep-summary.json").read_text())]:
    cut = summary.get("condition") or f"{summary['truncate']}-{summary['max_length']}"
    condition = f"{summary['domain']}/{summary['judge']}/{summary['semester']}/{cut}"
    if summary["status"] != "prepared":
        continue
    status = (CHANNELS_ROOT if cut.startswith("channels-") else ROOT) / "results" / condition / "status.json"
    state = json.loads(status.read_text()) if status.exists() else {"status": "not started"}
    rated = done = errors = 0
    for path in Path(state.get("log_dir", "/nonexistent")).glob("*.eval"):
        header = read_eval_log(path, header_only=True)
        if header.status == "error":
            errors += 1; continue
        done += header.status == "success"
        rated += sum(s.scores is not None and next(iter(s.scores.values())).answer is not None
                     for s in read_eval_log_sample_summaries(str(path)))
    print(json.dumps({"condition": condition, "status": state["status"], "new_ratings_expected": summary["new_ratings"],
                      "rated": rated, "tasks_done": done, "tasks_total": summary["tasks"], "error_logs": errors,
                      "valid_ratings": state.get("valid_ratings"), "metrics": state.get("metrics")}))
