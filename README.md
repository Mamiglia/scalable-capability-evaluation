# Capability estimation from saturated-benchmark transcripts

Code for the submission. Can a model's capability be estimated from its correctly answered transcripts on
*saturated* benchmark items, by protocols calibrated on earlier models and applied to newer ones?

Built on [Inspect AI](https://inspect.aisi.org.uk/). Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/).

## Install

```bash
uv sync                        # pinned by uv.lock; matharena is vendored under third_party/
uv run pytest                  # offline, no API calls
cp env.example .env            # API keys: only needed to re-collect or re-judge
```

## Data

Everything the paper uses is released as four Hugging Face datasets (anonymised for review):

| dataset | local path | contents |
| --- | --- | --- |
| `ANONYMOUS/benchmarks` | `data/` (git submodule) | every model's rollouts (`data/<bench>/<hash>/<model>/<repeat>/*.eval`) and the frozen tables (`data/meta/<domain>/`) |
| `ANONYMOUS/judging` | `logs/` | every pairwise and absolute-scoring judge run (`logs/judge/`, `logs/direct/`, and the `_channels-*`, thirds and reasoning-hidden conditions) |
| `ANONYMOUS/glm_features` | `glm_features/` | deterministic per-transcript features |
| `ANONYMOUS/transcript-embeddings` | `transcript-embeddings/` | Qwen3-Embedding-8B transcript vectors |

```bash
git submodule update --init data
uv run hf download ANONYMOUS/judging --repo-type dataset --local-dir logs
uv run hf download ANONYMOUS/glm_features --repo-type dataset --local-dir glm_features
uv run hf download ANONYMOUS/transcript-embeddings --repo-type dataset --local-dir transcript-embeddings
```

`SCE_DATA` and `SCE_EMBEDDINGS` point at copies elsewhere. With these downloaded, every analysis below runs from
the released logs and makes **no API calls**. Judging is collected on demand: a judge condition whose logs are
missing is judged through OpenRouter, which costs money.

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

**Protocols** (`src/protocols/`), with the paper's names:

| paper | code |
| --- | --- |
| Baseline (prediction from saturated items) | `easy_items` (`analysis/baselines.py`) |
| Deterministic transcript features | `glm_regex` (`protocols/transcript_features/`) |
| Transcript embeddings | `embedding` (`protocols/embeddings.py`) |
| Absolute scoring | `direct_judge_<judge>` (`protocols/direct_judge/`) |
| Pairwise judging | `judge_<judge>` (Bradley-Terry) and `judgelin_<judge>` (linear calibration) (`protocols/pairwise/`) |

**Metric.** Squared error between θ̂ and θ\* per target, and the log-likelihood ratio of θ\* under the
protocol's forecast against the baseline's (`src/analysis/challenges/likelihood_ratio.py`).

## Pipeline

```
uv run collect    rollouts of every model on every bench          -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run reference  freeze a domain: global 2PL, semester tasks, θ*  -> data/meta/<domain>/{models,items,responses}.csv
uv run evaluate   judging on demand, fit / predict every protocol  -> logs/judge/..., data/meta/<domain>/eval_<protocol>.csv
```

```bash
# collection (costs money; the released data already holds it): benches x models x repeats
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2
scripts/collect_openrouter.sh          # the sweeps: configs/collect/*.txt, one script per bench family

# freeze a domain (already frozen in data/meta)
uv run reference configs/domain/science.yml

# score protocols; --judge / --direct-judge add LLM judges (released logs are reused)
uv run evaluate configs/domain/science.yml --protocol ridge embedding
uv run evaluate configs/domain/science.yml --judge configs/judges/reasoning/kimi-k2-thinking.yml --semester 2026-I
uv run evaluate configs/domain/science.yml --direct-judge configs/judges/kimi-k2-thinking.yml --semester 2026-I
```

## Reproducing the paper

Run from the repository root. `results.py` comes first: it writes `analysis/results/forecasts.csv`, which the
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
