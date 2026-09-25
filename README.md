# Evaluating Model Capabilities After Benchmark Saturation

Code for the anonymous submission. Built on [Inspect AI](https://inspect.aisi.org.uk/); Python ≥ 3.13 and
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

| paper section | what it covers | code |
| --- | --- | --- |
| §2 Problem Setting | capability θ as a 2PL latent; saturated (EASY) vs held-out (HARD) items; semester tasks with references and targets | `src/irt/`, `src/reference/freeze.py` (item split, θ\*), `src/capability/task.py` (Task, Cohort), `configs/semesters.yml` |
| §3.1 Evaluation Setup | four domains, their benchmarks and models; rollout collection | `configs/domain/`, `configs/bench/`, `configs/models/`, `src/collect/` |
| §3.2 Prediction Protocols | baseline, deterministic features, embeddings, absolute scoring, pairwise judging | `src/protocols/`, `src/judge/`, `configs/judges/`, `src/analysis/baselines.py` |
| §4.1 Saturated transcripts contain capability information (Fig. 3) | squared error and likelihood-ratio tests per protocol | `scripts/analyses/results.py`, `scripts/analyses/likelihood_ratio.py`, `scripts/figures/manuscript_error_figures.py` |
| §4.2 Signal is distributed across the transcript: position (Fig. 4) | judging with thirds of each transcript removed | `src/challenges/thirds/`, `scripts/analyses/position_contributions.py` |
| §4.2 Signal is distributed across the transcript: channels (Fig. 5) | judging with reasoning, text, tool calls or tool outputs removed | `configs/judges/channels/`, `scripts/analyses/channel_contributions.py` |
| §4.3 Generalisation across time and domains (Fig. 6) | per-semester and per-domain error | `scripts/figures/manuscript_error_figures.py` |
| §4.3 Reasoning-model shift (Fig. 7) | pre-reasoning judges on reasoning models, reasoning hidden vs shown | `scripts/analyses/generalisation.py` |
| §4.3 More capable targets (Fig. 8) | error against distance above the strongest reference | `scripts/analyses/generalisation.py` |
| §4.3 Closed-source frontier models | frontier targets on the coding subset | `src/challenges/frontier/` |
| §4.3 Domain differences (Table) | per-domain error by protocol | `scripts/analyses/signal_glm.py`, `scripts/analyses/results.py` |
| Appendix | full methodology, configs, per-protocol ablations and robustness checks | see [Reproducing the paper](#reproducing-the-paper) |

## Install

```bash
uv sync                        # pinned by uv.lock; matharena is vendored under third_party/
uv run pytest                  # offline, no API calls
cp env.example .env            # API keys: only needed to re-collect or re-judge
```

### Data: the supplementary zip

The full data (every rollout and judge run, >100 GB) will be released after review. The supplementary material
holds one complete semester task, **cyber 2026-I** (`cyber-2026-I.zip`, 64 MB): everything the forecasting
pipeline reads for that task. From it every protocol's forecasts for the task's 8 target models are recomputed
exactly, with no API calls.

| path | contents |
| --- | --- |
| `data/meta/cyber/` | the frozen tables: item parameters, EASY/HARD split and train items, θ\* per task |
| `data/<bench>/<hash>/<model>/0/*.eval` | repeat-0 rollouts of the task's 8 references, 7 historical models and 8 targets on CyBench and InterCode, train items only |
| `logs/judge/cyber/2026-I/<judge>/24d6cc32/` | pairwise verdicts of the four judges, subjects' reasoning shown |
| `logs/direct/cyber/2026-I/<judge>/` | absolute-scoring ratings of the three judges |
| `glm_features/`, `transcript-embeddings/` | deterministic features and Qwen3-Embedding-8B vectors of those rollouts |

To fit the size limit the logs are reduced to what the analyses read: rollouts keep the train items without their
event traces, judge logs keep their verdict summaries (each judge's verdict and explanation, not the prompt).
Machine paths and account names in the logs are replaced by placeholders (`/home/user`, `/workspace/sce`).

**Reproducing the cyber 2026-I results**

```bash
# 1. install (see above), then unzip at the repository root: creates data/, logs/, glm_features/, transcript-embeddings/
unzip cyber-2026-I.zip

# 2. restrict every analysis to the released task, and make sure no judging can be launched
export SCE_DOMAINS=cyber SCE_SEMESTERS=2026-I
unset OPENROUTER_API_KEY

# 3. forecasts of every protocol, then the likelihood-ratio tests, then the tables and figures that use them
uv run python scripts/analyses/results.py            # -> analysis/results/forecasts.csv, metrics.csv, summary.csv
                                                     #    (its last cells need step 4's output and stop there on a first run)
uv run python scripts/analyses/likelihood_ratio.py   # -> analysis/results/likelihood_ratio.csv
uv run python scripts/analyses/results.py            # -> analysis/results/tables/, analysis/results/figures/
```

`results.py` prints the per-protocol squared error of the task; for cyber 2026-I:

| Saturated-item baseline | Deterministic features | Embeddings | Absolute scoring | Pairwise judge |
| ---: | ---: | ---: | ---: | ---: |
| 1.70 | 2.31 | 1.49 | 1.95 | **0.20** |

The analyses that pool domains or semesters (most figures), re-judging and re-embedding need the full data;
`scripts/export_subset.py <domain> <semester> --out <dir> --conditions 24d6cc32 --compact` builds the same archive
for any task from it. Without the data, everything can be recollected with the pipeline below (API costs apply).

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

In `src/protocols/`, with the paper's names:

| paper | code |
| --- | --- |
| Baseline (prediction from saturated items) | `easy_items` (`analysis/baselines.py`) |
| Deterministic transcript features | `glm_regex` (`protocols/transcript_features/`) |
| Transcript embeddings | `embedding` (`protocols/embeddings.py`) |
| Absolute scoring | `direct_judge_<judge>` (`protocols/direct_judge/`) |
| Pairwise judging | `judge_<judge>` (Bradley-Terry) and `judgelin_<judge>` (linear calibration) (`protocols/pairwise/`) |

## Pipeline and reproducing scripts

### Pipeline

```
uv run collect    rollouts of every model on every bench          -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run reference  freeze a domain: global 2PL, semester tasks, θ*  -> data/meta/<domain>/{models,items,responses}.csv
uv run evaluate   judging on demand, fit / predict every protocol  -> logs/judge/..., data/meta/<domain>/eval_<protocol>.csv
```

```bash
# collection (costs money): benches x models x repeats
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2
scripts/collect_openrouter.sh          # the sweeps: configs/collect/*.txt, one script per bench family

# freeze a domain (the zip ships data/meta/cyber frozen)
uv run reference configs/domain/science.yml

# score protocols; --judge / --direct-judge add LLM judges (existing logs are reused, missing ones judged)
uv run evaluate configs/domain/science.yml --protocol ridge embedding
uv run evaluate configs/domain/science.yml --judge configs/judges/reasoning/kimi-k2-thinking.yml --semester 2026-I
uv run evaluate configs/domain/science.yml --direct-judge configs/judges/kimi-k2-thinking.yml --semester 2026-I
```

### Reproducing the paper

These need the full data (the supplementary zip covers the cyber 2026-I rows only). Run from the repository
root. `results.py` comes first: it writes `analysis/results/forecasts.csv`, which the
other analyses read. Figures go under `analysis/`. The drawing code lives in `src/plots/`, one module per
analysis; `src/analysis/` and the scripts under `scripts/analyses/` compute.

| paper | command |
| --- | --- |
| all forecasts and per-task metrics | `uv run python scripts/analyses/results.py` |
| likelihood-ratio tests used by the figures and tables | `uv run python scripts/analyses/likelihood_ratio.py` |
| Fig. 3, Fig. 6, Table (time generalisation), appendix prediction example | `uv run python scripts/figures/manuscript_error_figures.py` |
| Fig. 4, appendix: transcript thirds | `uv run python scripts/analyses/position_contributions.py` (judge runs: `src/challenges/thirds/`) |
| appendix: thirds, other protocols | `uv run python scripts/analyses/other_protocol_thirds.py` |
| Fig. 5: transcript channels | `uv run python scripts/analyses/channel_contributions.py` (judges: `configs/judges/channels/`) |
| appendix: channels, other protocols | `uv run python scripts/analyses/other_protocol_channels.py` (inputs: `src/challenges/channels/`) |
| Fig. 7, Fig. 8: reasoning shift, capability distance | `uv run python scripts/analyses/generalisation.py` |
| frontier models | `src/challenges/frontier/` (see its README) |
| appendix: domain GLM effects and contrasts | `uv run python scripts/analyses/signal_glm.py` |
| appendix: cross-domain transfer | `uv run python scripts/analyses/cross_domain.py` |
| appendix: judge choice | `uv run python scripts/analyses/judge_choice.py` |
| appendix: number of items | `uv run python scripts/analyses/item_count_effect.py --both` |
| appendix: number of references | `uv run python scripts/analyses/reference_count_effect.py` |
| appendix: protocol cost | `uv run python scripts/analyses/protocol_cost.py` |
| appendix: judge ensemble | `uv run python scripts/analyses/judge_ensemble.py` |
| appendix: static references (prediction horizon) | `uv run python -m challenges.horizon.run build`, then `judge` and `transcript`, then `uv run python scripts/figures/prediction_horizon_figures.py` |
| appendix: data figures (accuracy, difficulty, θ by release) | `uv run python scripts/figures/domain_figures.py science math coding cyber` |
| deterministic features, embeddings (inputs) | `scripts/glm_features/extract.py`, `scripts/embeddings/` |

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
  capability/   Task, Cohort, and the evaluate CLI
  protocols/    the forecasting protocols
  judge/        judging prompt, dataset, scorer, eval_set runner
  configs/      config dataclasses and the two content hashes (data path, judge condition)
  analysis/     results tables, statistics, the research-question analyses (challenges/)
  plots/        every figure
  challenges/   harnesses for the thirds, channels, frontier and prediction-horizon experiments
scripts/
  analyses/     one script per analysis (compute, then draw with src/plots)
  figures/      scripts that only draw from saved results
third_party/    matharena (MIT), vendored
```
