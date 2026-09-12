# small-judge

Benchmark-free capability estimation. Can a model's capability be estimated from its
transcripts alone, without scoring it on a benchmark, and does that work when the estimator
is a weaker model, calibrated on last year's models and applied to this year's?

Built on [Inspect AI](https://inspect.aisi.org.uk/). Python ≥ 3.13, [`uv`](https://github.com/astral-sh/uv).

## Methodology

**Capability is the latent θ of an item response model.** Following Epoch AI / Artificial
Analysis and psychometrics, a model α and a benchmark item i interact through a 2PL with a
guessing floor:

    P(α solves i) = c_i + (1 − c_i) · σ(a_i (θ_α − b_i))

`a_i` is the item's discrimination, `b_i` its difficulty, `c_i` the guessing floor (1/n choices
on multiple choice, 0 otherwise). A benchmark score is one estimator of θ, and it saturates once
the items are easy for everyone; θ is what we care about. Capability is assumed unidimensional
within a *domain* (science, math, coding, cyber), so every domain is fit separately.

**Semester tasks.** Models are grouped by release semester (`configs/semesters.yml`, 8 models
each, 2023 as one semester). A *task* is one semester: the 8 models of the previous semester
are the **references**, whose θ\* is known, and the 8 models of this semester are the
**targets**, whose θ\* a protocol must forecast. Calibrating on the references and forecasting
the targets is exactly the old-to-new question.

**EASY, HARD, and θ\*.** For each task the 3 strongest references define the item split:
items all 3 solve on their first attempt are EASY, items all 3 answered but some failed are
HARD. A protocol sees only a fixed subset of the EASY items (the *train set*, 25 per difficulty
quartile) and only correct transcripts on them: nothing about what a model cannot do. The
ground truth θ\* is fit on the HARD items only, with `a, b, c` fixed to the global fit, so it is
capability beyond what the easy items measure, and it sits on one scale for every semester.

**Protocols.** A protocol is `fit(references) → predict(targets)`. Two families (or more) exist:

- *Tabular regression* on transcript statistics (length, answer length, reasoning length,
  output tokens): a linear fit of θ\* on the references, applied to the targets.
- *Pairwise judging*: an LLM judge compares a target's transcript with a reference's on the
  same item and picks the more capable model. Every (target, reference) pair is judged, in both
  shown orders. A Bradley-Terry model over the verdicts with the references **pinned at θ\***
  places each target directly on the θ scale.

**Metric.** Mean squared error between θ̂ and θ\* over the targets, per semester and pooled.
Forecasts are on the θ scale by construction, so no shift- or scale-free variant is reported.

## Pipeline

```
uv run collect    1. rollouts of every model on every bench          -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run reference  2. freeze a domain: global 2PL, semester tasks, θ*  -> data/meta/<domain>/{models,items,responses}.csv
                  3. Task.from_semester: reference + target cohorts on the train items
uv run evaluate   4. judging on demand (JudgeBT)                      -> logs/judge/<domain>/<semester>/<judge>/<hash>/*.eval
                  5. fit / predict every protocol, MSE vs θ*          -> data/meta/<domain>/eval_<protocol>.csv
```

```bash
uv pip install -e .                                   # source .env for OPENROUTER_API_KEY, RUNPOD, HF keys
uv run pytest

# 1. collect: benches × models × repeats (validate with --limit 5 --log-dir /tmp/smoke first; real runs cost money)
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2

# 2. freeze a domain once its collection is complete (refit with --force, then commit data/meta)
uv run reference configs/domain/science.yml

# 4+5. score protocols; --judge adds an LLM judge and runs the judging it is missing (costs money)
uv run evaluate configs/domain/science.yml --protocol chars ridge
uv run evaluate configs/domain/science.yml --judge configs/judges/gemma4-31b.yml --semester 2026-I
```

## Layout

```
configs/
  bench/        inspect tasks to collect (name = data dir, task = inspect spec, task_args)
  models/       one yml per model: OpenRouter slug + provider pin, or a Runpod vLLM block
  domain/       the benches that share one θ scale, with item filters and per-bench judge prompts
  judges/       an LLM judge and its condition (system prompt, reasoning shown, both orders, max length)
  collect/      model lists for collection sweeps (scripts/collect_*.sh)
  semesters.yml the roster: 8 models per release semester
src/
  collect/      stage 1: collection CLI, custom inspect tasks (collect/tasks/)
  model/        a ModelConfig -> eval_set kwargs; Runpod pod lifecycle (shared, ref-counted)
  evals/        reading .eval logs: one sample (sample.py), one log (transcripts.py), a domain (responses.py)
  irt/          encode a response table, fit 1PL / 2PL / fixed-item 2PL with NUTS
  reference/    stage 2: freeze.py fits and writes the frozen tables; tables.py reads them back
  capability/   stage 3 (task.py: Task, Cohort) and 5 (evaluate.py: forecast, score, CLI)
  protocols/    the methods: features.py (TabularRegression), pairwise/ (PairwiseBT, JudgeBT)
  judge/        stage 4: judging prompt and dataset, scorer, eval_set runner
  configs/      draccus config dataclasses and the two content hashes
data/           git submodule synced to Hugging Face: collected logs + frozen tables
logs/           judge runs, its own git repo synced to Hugging Face; logs/judge/ = one eval_set per <domain, semester, judge, condition>
scripts/        experiment helpers and analyses; nothing in src/ depends on them
```

Two content hashes decide where things live. `configs/paths.py:hash_task_args` hashes a bench's
resolved task arguments into the data path, so a changed argument is a new collection.
`configs/judge.py:condition_hash` hashes the judging condition into the judge run path.

## Configs

```yaml
# configs/bench/gpqa.yml — a bench
name: gpqa
task: inspect_evals/gpqa_diamond
task_args: {cot: true}

# configs/models/google/gemma4-31b.yml — a model (OpenRouter, one provider pinned; or a runpod: block)
name: google/gemma-4-31b-it
slug: openrouter/google/gemma-4-31b-it
release_date: '2026-03-31'
generate_args: {max_tokens: 235929, extra_body: {reasoning: {enabled: true}}}
model_args:
  provider: {only: [parasail/fp8], allow_fallbacks: false, require_parameters: true, quantizations: [fp8]}

# configs/domain/science.yml — a domain
name: science
benches:
  - bench: configs/bench/gpqa.yml
    system_prompt: assets/system_prompt/mcq.txt
  - bench: configs/bench/hle.yml
    hle_gold: true
    metadata: {category: [Physics, Chemistry, ...]}

# configs/judges/gemma4-31b.yml — a judge protocol
name: gemma4-31b
judge: configs/models/google/gemma4-31b.yml
include_reasoning: false
```

## Adding a protocol

Subclass `protocols.Protocol`, implement `fit(train: Cohort)` and `predict(test: Cohort) -> pd.Series`
(θ̂ indexed by target model, NaN where the protocol has nothing to say), and register an instance
in `protocols.PROTOCOLS`. A cohort exposes `models` (θ, log paths), `items` (the train items) and
`transcripts` (model → (bench, item) → the correct first-attempt transcript). A comparison-based
protocol subclasses `protocols.pairwise.PairwiseBT` and implements `verdicts(train, test)` only.

See `CLAUDE.md` for the architecture and `AGENTS.md` for operations: collection sweeps,
OpenRouter provider pinning, Runpod, resume semantics, data hygiene.
