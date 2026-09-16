#!/usr/bin/env python3
"""Which logs of a domain's frozen tables still lack embedding files.

    python scripts/embeddings/coverage.py configs/domain/science.yml [configs/domain/math.yml ...]

For every rostered model of data/meta/<domain>/models.csv (a role in some semester task: the only
models a protocol reads) and every path_<bench> column, checks that
<SMALL_JUDGE_EMBEDDINGS>/<path>.<variant>.npz exists for both variants and prints, per bench, how
many logs are covered plus an example missing path, so a fill pass can be targeted. Unrostered
rows are counted separately and never fail the check. Exit 1 while a rostered log is missing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from configs.domain import DomainConfig  # noqa: E402
from protocols.embeddings import EMBEDDINGS, VARIANTS, embedding_path  # noqa: E402
from reference import tables  # noqa: E402


def covered(path: str) -> bool:
    return all(embedding_path(path, v).exists() for v in VARIANTS)


def main() -> int:
    missing_total = 0
    for spec in sys.argv[1:] or ["configs/domain/science.yml"]:
        domain = DomainConfig.from_cli(spec)
        models, _ = tables.load(domain)
        rostered = models.filter(like="_role").notna().any(axis=1)
        print(f"== {domain.name}: {len(models)} models ({int(rostered.sum())} rostered), embeddings under {EMBEDDINGS}")
        for column in [c for c in models.columns if c.startswith("path_")]:
            paths = models.loc[rostered, column].dropna()
            missing = [p for p in paths if not covered(p)]
            other = [p for p in models.loc[~rostered, column].dropna() if not covered(p)]
            print(f"  {column[5:]:20s} {len(paths) - len(missing):4d}/{len(paths):<4d} rostered logs embedded"
                  + (f"; missing e.g. {missing[0]}" if missing else "")
                  + (f"; {len(other)} unrostered logs missing" if other else ""))
            missing_total += len(missing)
    print("complete" if not missing_total else f"{missing_total} rostered logs still lack embeddings")
    return 1 if missing_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
