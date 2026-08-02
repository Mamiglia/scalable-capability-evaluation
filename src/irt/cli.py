"""Fit IRT models on a response table, store abilities/difficulties, optionally validate."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import draccus
import pandas as pd

from irt.fitting import save_results
from irt.validation import compare


@dataclass
class IrtConfig:
    responses: str = "external_data/epoch-gpqa/responses.parquet"
    name: str = "gpqa"
    out_dir: str = "irt_results"
    fit_models: List[str] = field(default_factory=lambda: ["1pl", "2pl"])
    warmup: int = 1000
    samples: int = 1000
    validate: bool = False  # also run the held-out model comparison (slow)
    k: int = 5


def main() -> int:
    cfg = draccus.parse(IrtConfig)
    responses = pd.read_parquet(cfg.responses)
    caps, diffs = save_results(
        responses, models=tuple(cfg.fit_models), out_dir=cfg.out_dir,
        name=cfg.name, warmup=cfg.warmup, samples=cfg.samples,
    )
    print(f"stored {cfg.name}_capabilities.csv ({len(caps)} models), "
          f"{cfg.name}_difficulties.csv ({len(diffs)} items)")
    if cfg.validate:
        table = compare(responses, k=cfg.k, warmup=cfg.warmup, samples=cfg.samples)
        table.to_csv(Path(cfg.out_dir) / f"{cfg.name}_model_comparison.csv", index=False)
        print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
