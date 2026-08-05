# Small-Judge

`small-judge` is a lightweight Python framework built on top of [Inspect AI](https://inspect.ai-safety-institute.org.uk/) for running benchmark collection and pairwise LLM judge evaluations comparing candidate model pairs across benchmarks.


## 📁 Repository Structure

```text
small-judge/
├── configs/              # YAML configuration files
│   ├── bench/            # Benchmark task configs (e.g., gsm8k, mbpp, core-bench)
│   ├── models/           # Model configs (e.g., ling-2.6-flash)
│   └── experiments/      # Experiment configs combining benchmark, judge, major, & minor models
├── src/                  # Core library source code (DO NOT DIRECTLY EDIT FOR EXPERIMENTAL CHANGES)
│   ├── collect/          # Collection CLI implementation
│   │   └── cli.py
│   ├── configs/          # Dataclass configuration schemas (base, benchmark, model, experiment, utils)
│   │   ├── base.py
│   │   ├── task.py
│   │   ├── experiment.py
│   │   ├── model.py
│   │   └── utils.py
│   └── judge/            # Judge task builder, dataset constructor, scorer, & CLI
│       ├── cli.py
│       ├── dataset.py
│       ├── scorer.py
│       ├── task.py
│       └── util.py
├── script/               # Helper scripts (e.g., leave-one-out experiments)
│   └── run_corebench_leave_one_out.sh
├── tests/                # Pytest unit test suite
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_dataset.py
│   ├── test_scorer.py
│   ├── test_task.py
│   └── test_util.py
├── pyproject.toml        # Project dependencies and CLI entrypoints
└── README.md
```

---

## 🚀 Installation & Setup

Ensure you have [`uv`](https://github.com/astral-sh/uv) installed.

1. Clone the repository and install dependencies in editable mode:
```bash
uv pip install -e .
```

2. (Optional) Set up required environment variables (e.g. `OPENROUTER_API_KEY` or model provider API keys) in a `.env` file or shell environment.

---

## 🛠️ Usage

### 1. Benchmark Data Collection (`collect`)

The `collect` CLI evaluates one or more models on specified benchmarks and saves the `.eval` log files into `data/`.

```bash
# Run collection for a single benchmark and model
uv run collect -b configs/bench/gsm8k.yml -m configs/models/ling-2.6-flash.yml -n 1

# Run collection across multiple benchmarks and models
uv run collect \
  -b configs/bench/gsm8k.yml -b configs/bench/mbpp.yml \
  -m configs/models/ling-2.6-flash.yml \
  -n 1
```

Models can provision a temporary vLLM Pod by adding a `runpod` block. This
requires an installed and authenticated `runpodctl`:

```yaml
name: qwen/qwen3.5-9b
slug: openrouter/qwen/qwen3.5-9b
model_args:
  provider: {}
runpod:
  use: false
  model: Qwen/Qwen3.5-9B
  gpu_id: NVIDIA A40
  vllm_args:
    - --reasoning-parser
    - qwen3
    - --max-model-len
    - 32768
```

Set `use: true`, or force it for one collection with `--runpod`:

```bash
uv run collect --runpod \
  -b configs/bench/gsm8k.yml \
  -m configs/models/qwen3.5-9b.yml \
  -n 1
```

`Model` creates the Pod, waits for its OpenAI endpoint, and deletes it when the
last evaluation context exits, including on errors and interrupts. Concurrent
contexts with the same Runpod configuration share one Pod. It defaults to
Community Cloud, one GPU, a 40 GB container disk, four-hour auto-termination,
and vLLM prefix caching. Judge evaluations use the same lifecycle whenever
their judge model has `runpod.use: true`.

### 2. Pairwise Judge Evaluation (`judge`)

The `judge` CLI evaluates pairs of model outputs (Major model vs. Minor model) using a designated Judge model.

```bash
# Run a judge experiment from an experiment YAML config
uv run judge --config_path configs/experiments/core-bench/judge-gemma-easy.yml

# Or specify parameters directly via CLI flags
uv run judge \
  -b configs/bench/core-bench/easy.yml \
  --major google/gemma-4-31b-it \
  --minor qwen/qwen3.5-9b \
  -j openrouter/google/gemma-4-31b-it
```

---

## ⚙️ Configuration Format

### Benchmark Config (`TaskConfig`)
Example `configs/bench/gsm8k.yml`:
```yaml
name: inspect_evals/gsm8k
limit: 100
task_args:
  fewshot: 0
```

### Model Config (`ModelConfig`)
Example `configs/models/ling-2.6-flash.yml`:
```yaml
name: openrouter/inclusionai/ling-2.6-flash
slug: openrouter/inclusionai/ling-2.6-flash
model_args:
  provider: {}
```

### Experiment Config (`ExperimentConfig`)
Example `configs/experiments/core-bench/judge-gemma-easy.yml`:
```yaml
name: judge-gemma-easy
judge: openrouter/google/gemma-4-31b-it
major: moonshotai/kimi-k2.6
minor: qwen/qwen3.5-9b
benchmark: !include ../../bench/core-bench/easy.yml
mode: paired
```
