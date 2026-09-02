"""Print word counts / scores / providers for a smoke log dir."""

import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

for lp in sorted(Path(sys.argv[1]).rglob("*.eval")):
    log = read_eval_log(str(lp))
    model = lp.parent.parent.name
    wcs = [len((s.output.completion or "").split()) for s in log.samples]
    scores = "".join(s.scores["choice"].value if s.scores else "?" for s in log.samples)
    provs = {
        e.call.response.get("provider")
        for s in log.samples
        for e in s.events
        if e.event == "model" and e.call
    }
    print(f"{model:28} words={wcs} scores={scores} providers={provs}")
