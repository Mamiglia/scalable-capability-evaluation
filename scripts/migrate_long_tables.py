"""One-off (2026-10-09): the per-domain wide tables data/meta/<domain>/{models,items,responses}.csv -> the long
tables of reference.tables, coding_frontier folded into coding as the `frontier` cohort. Reshape only, no refit.

    uv run python scripts/migrate_long_tables.py <dir with the old <domain>/ tables>
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from configs.domain import DomainConfig
from evals.responses import eval_logs
from reference.tables import DATA, META, to_long, write

old = Path(sys.argv[1])
read = lambda d: (pd.read_csv(old / d / "models.csv", index_col="model", dtype={"semester": str}),
                  pd.read_csv(old / d / "items.csv", dtype={"item": str}),
                  pd.read_csv(old / d / "responses.csv", dtype={"item": str}))

for domain in ["science", "math", "coding", "cyber"]:
    models, items, responses = read(domain)
    logs = eval_logs(DomainConfig.from_cli(f"configs/domain/{domain}.yml"))
    logs["path"] = [str(Path(p).relative_to(DATA)) for p in logs.path]
    tables = to_long(models, items, responses, logs)
    if domain == "coding":  # + the frontier cohort: coding_frontier's 2026-I targets (src/challenges/frontier/integrate.py)
        cf_models, cf_items, cf_responses = read("coding_frontier")
        added = cf_models.index.difference(models.index)
        frontier = to_long(cf_models, cf_items, cf_responses[cf_responses.model.isin(added)], logs.iloc[:0], cohort="frontier")
        targets = frontier["tasks"].query("semester == '2026-I' and role == 'target' and model in @cf_models.index")
        targets = targets[~targets.model.isin(models.index[models["2026-I_role"].eq("target")])]
        tables["tasks"] = pd.concat([tables["tasks"], targets])
        tables["models"] = pd.concat([tables["models"], frontier["models"].query("model in @added")])
        tables["samples"] = pd.concat([tables["samples"], frontier["samples"]])
        fit = yaml.safe_load((old / "coding_frontier" / "fit.yml").read_text())
        base = yaml.safe_load((old / "coding" / "fit.yml").read_text())
        base["cohorts"] = {"frontier": {k: v for k, v in fit.items() if k not in base}}
        (META / "coding" / "fit.yml").write_text(yaml.safe_dump(base, sort_keys=False))
    for table, rows in tables.items():
        write(table, domain, rows)
    print(domain, {t: len(r) for t, r in tables.items()}, "| samples without a log:", int(tables["samples"].log.isna().sum()))
