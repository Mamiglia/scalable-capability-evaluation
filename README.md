# Small Judge

Research code for testing whether benchmark transcripts contain signal about
the capability of the model that produced them, even after conditioning on the
answer being correct.

The repository has two deliberately separate workflows:

```text
configs/bench/ ────────> collect ──> data/raw/ (append-only heap)
                                         │
                                         v
                                   data/dataset/ (soft-link tree)
                                         │
configs/experiments/ ──> judge ──────────┘
              │
              v
             logs/
```

Raw logs live in `data/raw/`. Code organizes and resolves logs from metadata
inside Inspect headers by soft-linking files into `data/dataset/` with the
structure `benchmark/task_args/model/{idx}.eval`.

## Collect benchmark transcripts

Benchmark settings live in `configs/bench/`. Models are selected independently
on the command line.

```sh
uv run collect \
  --bench configs/bench/gpqa.yml \
  --model configs/models/ling-2.6-flash.yml \
  --log_dir data/raw/runs/gpqa-ling-2.6-flash \
  --limit 2
```

Use a fresh `log_dir` for each eval set; Inspect rejects a directory
containing logs from a different eval set. Undeclared options such as
`--limit 2` are parsed as kwargs and forwarded to Inspect. GPQA is configured
with four epochs, so a limit of two runs two dataset samples per epoch.

For example, the GPQA benchmark manifest is:

```yaml
name: gpqa
task: inspect_evals/gpqa_diamond
log_dir: data/raw/gpqa
```

Models are supplied with a repeatable `--model` option. Each value may be a
model config path or a direct provider/model identifier:

```sh
uv run collect \
  --bench configs/bench/mbpp.yml \
  --model configs/models/model1.yml \
  --model provider2/model2
```

## Explore the data heap

Build soft links under `data/dataset/` from Inspect log headers in `data/raw/`:

```sh
uv run python src/registry/build.py
```

This constructs `data/dataset/benchmark/task_args/model/{idx}.eval`
pointing directly to the corresponding raw log in `data/raw/` sorted chronologically.
It is derived data and can be deleted and rebuilt at any time.

## Run judge experiments

Judge experiments compare existing successful collection logs and refresh the
dataset soft links automatically by default. Experiment manifests live in
`configs/experiments/`:

```yaml
name: gemma-31b-v-2b-gpqa
judge: openrouter/inclusionai/ling-2.6-flash
major: vllm/google/gemma-4-31B-it
minor: vllm/google/gemma-4-26B-A4B-it
benchmark: !include ../bench/gpqa.yml
task_args:
  cot: true
  epochs: 4
```

Run a two-sample experiment:

```sh
uv run judge \
  --config_path configs/experiments/gemma-gpqa.yml \
  --limit 2
```

Declaring fields and arbitrary Inspect kwargs can both be overridden:

```sh
uv run judge \
  --config_path configs/experiments/gemma-gpqa.yml \
  --include_reverse false \
  --limit 20
```

Judge results are written to `logs/`. The source paths and experiment controls
are recorded in the Inspect task metadata.

## Code layout

```text
src/
├── collect/     # benchmark data collection
├── registry/    # soft-link builder and path resolver over data/raw -> data/dataset
└── judge/       # transcript pairing, judging, scoring, experiment CLI
```
