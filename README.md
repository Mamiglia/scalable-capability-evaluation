<div align="center">

# Evaluating Model Capabilities After Benchmark Saturation

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Inspect AI](https://img.shields.io/badge/Built%20with-Inspect%20AI-4B6EAF)](https://inspect.aisi.org.uk/)
[![Package manager](https://img.shields.io/badge/Package%20manager-uv-purple)](https://docs.astral.sh/uv/)
[![Data](https://img.shields.io/badge/Data-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/generality-labs)

</div>


This is the official repository for the paper _Evaluating Model Capabilities After Benchmark Saturation_.


Authors:
[Morio Hamada*](https://www.linkedin.com/in/morio-hamada/),
[Jehyeok Yeon*](https://jeybird248.github.io/),
[Matteo Migliarini*](https://mamiglia.github.io/#/)
[Justin Olive](https://www.linkedin.com/in/justindanielolive/)
[Magda Dubois](https://www.linkedin.com/in/magdadubois/)
(* Equal Contribution)
---

## Overview

Rapid advancements in AI capabilities are causing benchmarks to saturate. Once several models approach perfect
scores, a benchmark loses its power to differentiate between them, and with it its utility for informing AI safety
cases and frontier policies. Yet benchmark items produce far more information than the scores we typically extract
from them: a full transcript of *how* each model solved the task. We investigate whether these transcripts can be
used to assess model capabilities, even after benchmark scores have saturated. We simulate saturation
retrospectively using historical benchmark and model releases, and test whether protocols that access only
correctly solved transcripts from saturated benchmarks can predict capabilities measured on held-out, unsaturated
items. We evaluate representative contemporary approaches for extracting information from model transcripts:
pairwise judging, single-transcript absolute scoring, and predictions from deterministic transcript features or
embeddings. We find that saturated transcripts retain capability information beyond benchmark accuracy. Amongst our
evaluated protocols, pairwise judging performs best, reducing prediction error by 5.27× compared to a baseline
using only saturated benchmark scores. This capability-relevant information is distributed across transcript
sections, but concentrated in specific channels of information such as reasoning and tool-calls. However, the
predictive power of protocols does not always generalise across distribution shifts -- for example, by
underestimating the capabilities of the first reasoning models introduced in early 2025. Saturated benchmarks thus
retain capability-predicting signal in their task transcripts, but extending the lifetime of existing benchmarks
requires methods designed to extract these signals more robustly.

This repository includes the evaluation data, protocols, analysis scripts, and a **plug-in interface for testing new capability-prediction methods**.

---

## Getting Started

### 1. Installation

Requires **Python 3.13 or later** and [`uv`](https://docs.astral.sh/uv/). Run all commands from the repository root.

```bash
git clone --recurse-submodules <this-repo-url>
cd <repository-directory>
uv sync
```

The `data/` submodule contains the [`generality-labs/benchmarks`](https://huggingface.co/datasets/generality-labs/benchmarks) dataset and uses Git LFS. Both published datasets are **gated**. Accept their Hugging Face terms before downloading. They prohibit training on the data and reposting benchmark questions in plain text.

### 2. Download data and run tests

```bash
uv run python scripts/analysis/fetch_data.py
uv run pytest
```

The data download populates `data/` and `logs/`; the full reproduction dataset is approximately **40 GB**. Tests run offline and do not make API calls.

### 3. Configure API keys (optional)

```bash
cp env.example .env
```

Only needed when collecting new model rollouts or running new LLM judges. Reproducing the published analyses and evaluating existing protocol outputs does **not** require paid API calls.

---

## Results

The main comparison covers **64 target models** across science, maths, coding, and cyber tasks in semesters `2025-II` and `2026-I`. Lower mean squared error (MSE) is better.

| Protocol | Coding | Cyber | Maths | Science | Overall MSE |
|:--|--:|--:|--:|--:|--:|
| **Pairwise judging** | **0.14** | **0.26** | **0.42** | **0.21** | **0.26** |
| Deterministic transcript features | 0.70 | 1.57 | 0.76 | 0.43 | 0.86 |
| Transcript embeddings | 1.69 | 1.29 | 0.66 | 0.40 | 1.01 |
| Absolute scoring | 1.41 | 1.48 | 0.85 | 0.31 | 1.01 |
| Saturated-score baseline | 2.14 | 1.39 | 1.32 | 0.65 | 1.37 |

These values come from the following offline command:

```bash
uv run evaluate --protocol pairwise direct glm_regex embedding mean_ref --semester 2025-II 2026-I
```

**Important limitation:** Signals that work well on familiar model cohorts may fail under distribution shift. In particular, some protocols underestimated the capabilities of early reasoning models in 2025.

---

## Usage: Evaluate a New Protocol

The repository provides a common interface to compare a new method with the methods in the paper, using the same saturated transcripts and held-out capability targets.

### 1. Run the existing protocols

```bash
uv run evaluate --protocol pairwise direct glm_regex embedding mean_ref --semester 2025-II 2026-I
```

The output reports MSE by domain, overall MSE, and the number of targets with a prediction (`n`). Forecasts are written to `results/<domain>/eval_<name>.csv`.

### 2. Add your own method

See [`scripts/example_protocol.py`](scripts/example_protocol.py) for a complete working example based on judge win rates against reference models (overall MSE: **0.57**).

A protocol implements `protocols.Protocol` and returns a `pd.Series` of capability predictions (`theta_hat`), indexed by target model:

```python
__call__(task) -> pd.Series
```

By default, the interface fits on `task.references` and predicts for `task.test`. Return `NaN` for targets the method cannot score.

```bash
uv run evaluate --protocol pairwise path/to/mine.py:Mine
```

### 3. Understand the task interface

| Field | Description |
|:--|:--|
| `task.domain` | Science, maths, coding, or cyber |
| `task.semester` | Evaluation semester |
| `task.references` | Previous semester's models, with known capability `θ*` |
| `task.historical` | Earlier models, with capability estimates on this task |
| `task.test` | Target models; capability labels are hidden |
| `task.truth` | Ground truth used by the scorer only; **do not use for prediction** |

Each cohort provides `models` (capability and `path_<bench>` log columns), `items` (benchmark, item, guessing floor, discrimination, and difficulty), and `transcripts` (correct first-attempt transcripts keyed by model and item).

**Evaluation rule:** Use only the fixed **100 training items per task**. The evaluated protocols all receive the same items, and must predict capability on held-out HARD items without using target labels.

---

## Methodology

### Capability estimation

We define capability as the latent model ability `θ` in a two-parameter logistic item response model with an item-specific guessing floor:

\[
P(\text{model }\alpha\text{ solves item }i)
= c_i + (1-c_i)\,\sigma\!\left(a_i(\theta_\alpha-b_i)\right).
\]

Here `a_i` is item discrimination, `b_i` is item difficulty, and `c_i` is the guessing floor. The model is fitted separately for **science, maths, coding, and cyber**.

### Historical semester tasks

Models are grouped into release semesters (eight models per semester; see [`configs/semesters.yml`](configs/semesters.yml)). For each task:

1. The previous semester's eight models form the **reference cohort**. Their `θ*` values are known.
2. The current semester's eight models form the **target cohort**. Their `θ*` values must be predicted.
3. Older models form the **historical cohort**.
4. The three strongest reference models determine the item split:
   - **EASY (saturated):** all three answer correctly on the first attempt.
   - **HARD (unsaturated):** all three answered, but at least one failed.
5. Protocols see a fixed stratified sample of **100 EASY items** (25 per difficulty quartile) and **only correct transcripts**. Ground-truth `θ*` is fitted using HARD items, with item parameters fixed to the global fit.

We evaluate squared error between predicted `θ̂` and ground-truth `θ*` for each target. The paper also reports log-likelihood ratios relative to the baseline (see `src/analysis/challenges/likelihood_ratio.py`).

### Evaluated protocols

| Method | Registered protocol | Implementation |
|:--|:--|:--|
| Saturated-score baseline | `mean_ref` | `src/protocols/baseline.py` |
| Deterministic transcript features | `glm_regex` | `src/protocols/transcript_features/` |
| Transcript embeddings | `embedding` | `src/protocols/embeddings.py` |
| Absolute scoring (1–10) | `direct` | `src/protocols/judges.py`, `src/protocols/direct_judge/` |
| Pairwise judging | `pairwise` | `src/protocols/judges.py`, `src/protocols/pairwise/` |

Pairwise judgments are converted to capability estimates using per-benchmark weighted least squares on pairwise logits. Absolute scores use a linear mapping from mean rating to capability. For each task, the judge with the lowest leave-one-out error on reference models is selected. Additional variants include `pairwise_calibrated`, `pairwise_bt`, and `direct_item_z`.

---

## Datasets

Both datasets are hosted under [Generality Labs on Hugging Face](https://huggingface.co/generality-labs). Access requires acceptance of the dataset terms.

| Dataset | Local location | Contents |
|:--|:--|:--|
| [`benchmarks`](https://huggingface.co/datasets/generality-labs/benchmarks) | `data/` | Model rollouts (`.eval`) and frozen reference tables |
| [`sce-protocols`](https://huggingface.co/datasets/generality-labs/sce-protocols) | `logs/` | Pairwise and absolute judge outputs, deterministic features, and embeddings |

Model rollouts live at `data/<bench>/<hash>/<model>/<repeat>/*.eval`. Protocol logs are organized by transcript view, such as `full`, `reasoning-hidden`, `thirds-<kept>`, and `channels-<only|drop>-<channel>`. See `logs/README.md` and `logs/CONTRACT.md` for the exact layouts.

### Frozen metadata tables

Tables in `data/meta/*.csv` cover the models and tasks in the paper. `data/meta/full/` also includes additional models and targets used for the global item fit. The table schema is defined in `src/reference/tables.py`; see `data/CONTRACT.md` for data rules.

| File | Contents |
|:--|:--|
| `samples.csv` | Per-rollout results, correctness, and log path |
| `items.csv` | Item parameters (`a`, `b`, `c`) and uncertainties |
| `models.csv` | Models, release semesters, and global capability estimates |
| `tasks.csv` | Semester cohort membership, roles, and task-specific `θ*` |
| `task_items.csv` | EASY/HARD splits, training flags, and difficulty quartiles |
| `judgments_pairwise.csv` | Individual pairwise verdicts and transcript conditions |
| `judgments_direct.csv` | Individual 1–10 ratings and transcript conditions |
| `forecasts.csv` | Published protocol predictions, target values, and metadata |

Judgment tables include the verdicts of each task's three strongest reference judges. The `core` flag identifies the scored run for each judge; `paper` identifies the run used for the paper's headline results.

---

## Reproduce the Paper

All published analysis steps use downloaded data and run **without new judging or API costs**.

```bash
# Fetch published transcripts, protocol outputs, features, and embeddings (~40 GB)
uv run python scripts/analysis/fetch_data.py

# Generate the paper's analyses (~3 hours)
uv run python scripts/analysis/run.py
```

To inspect available stages or resume a partial run:

```bash
uv run python scripts/analysis/run.py --list
uv run python scripts/analysis/run.py --from <step>
```

All analysis settings are in `scripts/analysis/config.yaml`. Output is written to `analysis/`, including protocol forecasts, likelihood ratios, figures, and CSVs containing the displayed numbers and corresponding tests.

| Analysis stage | Figures | Tables |
|:--|:--|:--|
| `capability_signal` | 3, 10, 16, 17, 21 | 3, 6, 9 |
| `transcript_parts` | 4, 5, 22, 23, 24 | |
| `generalisation` | 6, 7, 8, 9, 18 (cross-domain values) | 7 |
| `sensitivity` | 15, 19, 20, 25, 27, 28 | 8, 10 |

Additional figure commands:

```bash
# Domain data figures (11–14)
uv run python scripts/figures/domain_figures.py science math coding cyber

# Figure 26: prediction horizon (build, judge, transcript, then plot)
uv run python -m challenges.horizon.run build
# Run the 'judge' and 'transcript' horizon stages as documented by the CLI.
uv run python scripts/figures/prediction_horizon_figures.py
```

For the exact file generated by each stage, consult the docstring in `scripts/analysis/run.py`.

---

## Collect New Rollouts and Judgments (Optional)

These commands run models or judges and **may incur API or compute costs**. Set credentials in `.env` first.

### Collection

```bash
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2

# Predefined model and benchmark sweeps
bash scripts/collect_openrouter.sh
```

Collection writes Inspect AI rollout logs to `data/<bench>/<hash>/<model>/<repeat>/*.eval`.

### Build reference tasks

Once the domain's collection is complete, fit the domain-wide item response model and freeze its semester tasks:

```bash
uv run reference configs/domain/science.yml
```

This writes the full metadata to `data/meta/full/*.csv` and the paper subset to `data/meta/*.csv`.

### Run a new judge and evaluate it

```bash
uv run judge configs/domain/science.yml \
  --judge configs/judges/reasoning/kimi-k2-thinking.yml \
  --semester 2026-I

uv run evaluate configs/domain/science.yml \
  --judge configs/judges/reasoning/kimi-k2-thinking.yml \
  --semester 2026-I
```

The judging runner resumes unfinished runs and reads completed ones. To export judgment and forecast tables, use `scripts/export_judgments.py`.

### Pipeline summary

```text
collect   -> data/<bench>/<hash>/<model>/<repeat>/*.eval
reference -> data/meta/full/*.csv and data/meta/*.csv
judge     -> logs/pairwise/<view>/... or logs/absolute/<view>/...
evaluate  -> results/<domain>/eval_<protocol>.csv
analysis  -> analysis/ (figures, tables, forecasts and tests)
```

---

## Repository Structure

```text
configs/
  bench/           Inspect AI benchmark tasks
  models/          OpenRouter / Runpod vLLM model configurations
  domain/          Domain definitions, filters, judge prompts
  judges/          Judge and transcript-view configurations
  collect/         Model sweep lists
  semesters.yml   Release-semester model cohorts

src/
  collect/         Collection CLI and Inspect AI tasks
  model/           Model configuration and Runpod lifecycle
  evals/           Read and score .eval logs
  irt/             1PL / 2PL / fixed-item models with NUTS
  reference/       Fit, freeze, and load metadata tables
  capability/      Task / Cohort and evaluation CLI
  protocols/       Capability-prediction methods
  judge/           Prompting and judging pipeline
  configs/         Configuration types and content hashes
  analysis/        Paper analyses and statistics
  plots/           Shared plotting utilities
  challenges/      Thirds, channels, frontier and horizon studies

scripts/
  analysis/        Published analysis runner and data download
  figures/         Data and horizon figures
  judge/, audit/   Data provenance and verification
  embeddings/      Embedding extraction
  glm_features/    Deterministic feature extraction

data/              Rollouts and reference tables (Hugging Face)
logs/              Judge logs, features and embeddings (Hugging Face)
```

---

## Citation

If you use the code or datasets in research, please cite the paper. (citation coming soon!)

