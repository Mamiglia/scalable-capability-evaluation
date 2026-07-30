# Small Judge

Research code for testing whether benchmark transcripts contain signal about
the capability of the model that produced them, even after conditioning on the
answer being correct.

The repository has two deliberately separate workflows:

```text
configs/bench/ ──> collect ──> data/ (raw, append-only heap)
                                         │
                                         v
                                  data/registry.csv
                                         │
configs/experiments/ ──> judge ──────────┘
              │
              v
artifacts/experiments/<name>/<fingerprint>/
```

`data/` is source material, not an organizational scheme. Code discovers logs
from metadata inside the Inspect files through a regenerable registry.

## Collect benchmark transcripts

Collection uses composable draccus YAML files. Shared settings live in
`configs/base.yaml`; benchmark-specific settings live in
`configs/bench/`.

```sh
uv run collect \
  bench=core-bench \
  models=ling-2.6-flash \
  bench.task_args.difficulty=hard limit=50
```

Model-specific overrides can be placed in `configs/models/`. Collection output
goes under `data/`.

## Explore the data heap

Build the registry from Inspect log headers:

```sh
uv run python src/registry/build.py
```

The registry records the source path, eval ID, status, full model name,
benchmark, dataset, task arguments, creation time, and sample count. It is
derived data and can be deleted and rebuilt at any time.

## Run judge experiments

Judge experiments live in `configs/experiments/`:

```yaml
name: gemma-31b-v-2b-gpqa
judge_model: openrouter/inclusionai/ling-2.6-flash
major_model: gemma_31b
minor_model: gemma_2b
benchmark_name: gpqa_diamond

require_both_correct: true
include_reverse: true
max_length: 65536
limit: 100000
```

Run the experiment:

```sh
uv run judge configs/experiments/gemma-gpqa.yaml
```

Command-line overrides are accepted after the config path:

```sh
uv run judge configs/experiments/gemma-gpqa.yaml limit=20 include_reverse=false
```

Every resolved configuration and pair of source logs receives a stable
fingerprint. Results are stored in:

```text
artifacts/experiments/<name>/<fingerprint>/
├── manifest.json
└── logs/
```

This keeps raw collection logs separate from derived judge results and records
the exact inputs and experimental controls for every run. `run.sh` remains as
a compatibility shim for `uv run judge`.

## Code layout

```text
src/
├── collect/     # benchmark data collection
├── registry/    # metadata index over data/
└── judge/       # transcript pairing, judging, scoring, experiment CLI
```
