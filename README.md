# Evaluating Model Capabilities After Benchmark Saturation

Code and data for the paper. Built on [Inspect AI](https://inspect.aisi.org.uk/); Python ≥ 3.13 and
[`uv`](https://docs.astral.sh/uv/).

## Abstract

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

## The paper and the code

| paper | code |
| --- | --- |
| §2 Problem setting: θ as a 2PL latent, saturated (EASY) and held-out (HARD) items, semester tasks | `src/irt/`, `src/reference/` (item split, θ\*, the tables), `src/capability/task.py`, `configs/semesters.yml` |
| §3.1 Evaluation setup: domains, benchmarks, models, rollouts | `configs/domain/`, `configs/bench/`, `configs/models/`, `src/collect/` |
| §3.2 Prediction protocols | `src/protocols/`, `src/judge/`, `configs/judges/` |
| §4 and the appendix: every figure and table | `scripts/analysis/run.py` (below); code in `src/analysis/<question>/` |
| Closed-source frontier targets: collection and θ\* | `src/challenges/frontier/` |
| Thirds, channels and prediction-horizon judge runs | `src/challenges/{thirds,channels,horizon,truncation}/` |

## Install

```bash
git clone --recurse-submodules <this repo>   # data/ is the generality-labs/benchmarks dataset (git + LFS)
uv sync                                      # installs src/ as packages; run everything from the repository root
uv run pytest                                # offline, no API calls
cp env.example .env                          # API keys: only needed to collect or judge
```

## Data

Everything is public on Hugging Face under [generality-labs](https://huggingface.co/generality-labs):

| dataset | local path | contents |
| --- | --- | --- |
| `benchmarks` | `data/` (submodule) | every rollout (`data/<bench>/<hash>/<model>/<repeat>/*.eval`) and the frozen tables in `data/meta/` |
| `judging` | `logs/` | every judge run (`logs/judge/`, `logs/direct/`, ... as `.eval`) |
| `glm_features` | `glm_features/` | deterministic transcript features, one CSV per rollout log |
| `transcript-embeddings` | `transcript-embeddings/` | Qwen3-Embedding-8B vectors, one `.npz` per rollout log |

The tables in `data/meta/` are all a protocol needs besides the transcripts (schema: `src/reference/tables.py`,
rules: `data/CONTRACT.md`). `data/meta/*.csv` covers the paper; `data/meta/full/` adds the models and targets
outside it (the global item fit also used them).

| table | one row per | columns |
| --- | --- | --- |
| `samples.csv` | scored rollout | domain, bench, item, model, repeat, correct, log (its `.eval` under `data/`) |
| `items.csv` | item | domain, bench, item, a, a_sd, b, b_sd, c (global 2PL) |
| `models.csv` | model | domain, model, semester (release), theta_global, theta_global_sd |
| `tasks.csv` | model in a semester task | domain, semester, model, role (reference / historical / target), cohort, theta (θ\*), theta_sd |
| `task_items.csv` | item in a semester task | domain, semester, bench, item, split (easy / hard), train, quartile |
| `judgments_pairwise.csv` | pairwise verdict | domain, semester, judge, condition, bench, item, model_left, model_right, pick_left, chars_left, chars_right, core, paper |
| `judgments_direct.csv` | 1-10 rating | domain, semester, judge, condition, bench, item, model, score, epoch, core, paper |
| `forecasts.csv` | the paper's θ̂ per target | domain, semester, protocol, variant, judge, condition, judge_loo_mse, model, theta_hat, theta, missing, paper |

The judgment tables hold every verdict and rating of each task's judges (its three strongest references): `core`
marks the run per judge that the paper scores, `paper` the one behind the headline numbers.

## Method

**Capability is the latent θ of an item response model.** A model α and a benchmark item i interact through a
2PL with a guessing floor, `P(α solves i) = c_i + (1 − c_i) · σ(a_i (θ_α − b_i))`, fit per domain (science,
maths, coding, cyber).

**Semester tasks.** Models are grouped by release semester (`configs/semesters.yml`, 8 models each). In a task
the previous semester's 8 models are the **references**, whose θ\* is known, and this semester's 8 are the
**targets**, whose θ\* a protocol must forecast. Earlier models form the **historical** cohort.

**EASY, HARD and θ\*.** The 3 strongest references define the item split: items all 3 solve on their first
attempt are EASY (saturated), items all 3 answered but some failed are HARD. A protocol sees only a fixed
stratified subset of the EASY items (25 per difficulty quartile) and only correct transcripts on them. The
ground truth θ\* is fit on the HARD items only, with the item parameters fixed to the global fit.

**Metric.** Squared error between θ̂ and θ\* per target, and the log-likelihood ratio of θ\* under the
protocol's forecast against the baseline's (`src/analysis/challenges/likelihood_ratio.py`).

## Protocols

In `src/protocols/`, registered in `protocols.PROTOCOLS` under these names:

| paper | code |
| --- | --- |
| Baseline (prediction from saturated items) | `mean_ref` (`protocols/baseline.py`) |
| Deterministic transcript features | `glm_regex` (`protocols/transcript_features/`) |
| Transcript embeddings | `embedding` (`protocols/embeddings.py`) |
| Absolute scoring | `direct` (`protocols/judges.py`, from `judgments_direct.csv`; one judge's logs: `protocols/direct_judge/`) |
| Pairwise judging | `pairwise` (`protocols/judges.py`, from `judgments_pairwise.csv`; one judge's logs: `protocols/pairwise/`) |

`pairwise` and `direct` convert each judge's core run (pairwise: per-bench weighted least squares on pair logits;
direct: a line from mean rating to θ\*) and keep, per task, the judge with the lowest leave-one-out error on the
references. Variants: `pairwise_calibrated`, `pairwise_bt`, `direct_item_z`.

## Compete: a new protocol

`uv run evaluate` recomputes the paper's protocols offline (about 2 minutes) and scores any other protocol next to
them. The score is the MSE of θ̂ against θ\* over the 64 targets of the main tasks (2025-II and 2026-I, all four
domains); `n` counts the targets with a forecast.

```
$ uv run evaluate --protocol pairwise direct glm_regex embedding mean_ref --semester 2025-II 2026-I
domain     coding  cyber  math  science  main   n   all
protocol
pairwise     0.14   0.26  0.42     0.21  0.26  64  0.26
glm_regex    0.70   1.57  0.76     0.43  0.86  64  0.86
embedding    1.69   1.29  0.66     0.40  1.01  64  1.01
direct       1.41   1.48  0.85     0.31  1.01  64  1.01
mean_ref     2.14   1.39  1.32     0.65  1.37  64  1.37
```

A protocol is a `protocols.Protocol`: `__call__(task) -> pd.Series` of θ̂ indexed by target (NaN where it has
nothing to say), by default `fit(task.references)` then `predict(task.test)`. Write it in any file and score it with
`uv run evaluate --protocol pairwise path/to/mine.py:Mine`; `scripts/example_protocol.py` (judges' win rate
against the references, 0.57) is a complete one. A `Task` holds `domain`, `semester`, `references` (previous
semester, θ = θ\*), `historical` (older models, θ = this task's θ\*), `test` (targets, θ hidden) and `truth`
(scoring only). A cohort exposes `models` (θ, `path_<bench>` logs), `items` (the train items: bench, item, guess,
a, b) and `transcripts` (model → (bench, item) → the correct first-attempt transcript). Use only the train items:
every protocol in the paper sees the same 100 items per task. Forecasts go to `results/<domain>/eval_<name>.csv`.

## Pipeline and reproducing scripts

### Pipeline

```
uv run collect    rollouts of every model on every bench          -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run reference  freeze a domain: global 2PL, semester tasks, θ*  -> data/meta/full/*.csv (+ the paper subset, data/meta/*.csv)
uv run judge      collect a judge's verdicts or ratings            -> logs/judge/... , logs/direct/...
                  scripts/export_judgments.py                      -> data/meta/judgments_*.csv, forecasts.csv
uv run evaluate   fit / predict protocols, MSE vs θ* (never judges) -> results/<domain>/eval_<protocol>.csv
```

```bash
# collection (costs money): benches x models x repeats
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2
scripts/collect_openrouter.sh          # the sweeps: configs/collect/*.txt, one script per bench family

# freeze a domain once its collection is complete
uv run reference configs/domain/science.yml

# judge (costs money; resumes, a finished run is only read), then score that judge's run
uv run judge configs/domain/science.yml --judge configs/judges/reasoning/kimi-k2-thinking.yml --semester 2026-I
uv run evaluate configs/domain/science.yml --judge configs/judges/reasoning/kimi-k2-thinking.yml --semester 2026-I
```

### Reproducing the paper

The paper's figures and tables come from `scripts/analysis/run.py`, from the published datasets; nothing is judged or
paid for.

```bash
uv run python scripts/analysis/fetch_data.py         # transcripts, judge logs, features, embeddings (~40 GB)
uv run python scripts/analysis/run.py                # about three hours; --list shows the steps, --from resumes
```

Every setting is in `scripts/analysis/config.yaml`. Outputs land in `analysis/`: `forecasts/` (every protocol's
forecast of every target, and the log likelihood ratios), then one folder per question with its figures and a CSV of
every number they show (with its test):

| step | figures and tables |
| --- | --- |
| `capability_signal` | Fig. 3, 10, 16, 17, 21; Tables 3, 6, 9 |
| `transcript_parts` | Fig. 4, 5, 22, 23, 24 |
| `generalisation` | Fig. 6, 7, 8, 9, the numbers of Fig. 18 (cross-domain); Table 7 |
| `sensitivity` | Fig. 15, 19, 20, 25, 27, 28; Tables 8, 10 |

`run.py`'s docstring names the output behind each. Not in `run.py`: the data figures 11 to 14
(`scripts/figures/domain_figures.py science math coding cyber`) and Fig. 26, the prediction horizon
(`uv run python -m challenges.horizon.run build`, then `judge` and `transcript`, then
`scripts/figures/prediction_horizon_figures.py`).

## Layout

```
configs/
  bench/        inspect tasks to collect (name = data dir, task = inspect spec, task_args)
  models/       one yml per model: OpenRouter slug + provider pin, or a Runpod vLLM block
  domain/       the benches that share one θ scale, with item filters and per-bench judge prompts
  judges/       an LLM judge and its condition (reasoning/: shown, hidden/: hidden, channels/, direct/)
  collect/      model lists for the collection sweeps (scripts/collect_*.sh)
  semesters.yml the roster: 8 models per release semester
src/
  collect/      collection CLI and custom inspect tasks
  model/        a ModelConfig -> eval_set kwargs; Runpod pod lifecycle
  evals/        reading .eval logs: one sample, one log, a domain's responses
  irt/          1PL / 2PL / fixed-item 2PL with NUTS
  reference/    freeze.py fits and writes the frozen tables; tables.py reads them back
  capability/   Task, Cohort, and the evaluate CLI (scorer and leaderboard)
  protocols/    the forecasting protocols
  judge/        judging prompt, dataset, scorer, eval_set runner, the judge CLI
  configs/      config dataclasses and the two content hashes (data path, judge condition)
  analysis/     the paper's analyses: forecasts and statistics, one subpackage per question
  plots/        figure code shared by the analyses
  challenges/   harnesses for the thirds, channels, frontier and prediction-horizon experiments
scripts/
  analysis/     run.py and one runner per step, config.yaml, fetch_data.py
  figures/      the data figures and the prediction-horizon figure
  judge/, audit/  provenance: the imports, rescores and checks the published data went through
  embeddings/, glm_features/  the embedding and feature inputs
```
