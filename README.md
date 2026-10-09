# small-judge

Benchmark-free capability estimation. Can a model's capability be estimated from its
transcripts alone, without scoring it on a benchmark, and does that work when the estimator
is a weaker model, calibrated on earlier models and applied to this semester's?

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
**targets**, whose θ\* a protocol must forecast. Models from semesters before the references
form the **historical** cohort. `Task.references` and `Task.historical` are disjoint;
`Task.train` returns their union for supervised fitting. Only models with this task's θ\*
can enter the historical cohort. Current, future, and unrostered models are excluded.

**EASY, HARD, and θ\*.** For each task the 3 strongest references define the item split:
items all 3 solve on their first attempt are EASY, items all 3 answered but some failed are
HARD. A protocol sees only a fixed subset of the EASY items (the *train set*, 25 per difficulty
quartile) and only correct transcripts on them: nothing about what a model cannot do. The
ground truth θ\* is fit on the HARD items only, with `a, b, c` fixed to the global fit, so it is
capability beyond what the easy items measure, and it sits on one scale for every semester.
Older models receive their own θ\* on this task's HARD items using the same global item
parameters, in a separate fit that preserves the existing reference/target fit. Models with
no HARD responses are reported and left unlabelled; no prior-only estimates or another
semester's labels are substituted. Every cohort uses the same selected EASY items and
correct repeat-0 transcripts. A cohort's `semester` identifies the task, not a release group.

**Protocols.** Supervised protocols use `fit(task.train) → predict(task.test)`:

- *Tabular regression* on transcript statistics (length, answer length, reasoning length,
  output tokens): preprocessing and regression coefficients fit on references + history,
  applied to the targets. Models with missing features are excluded from fitting/prediction.
- *Pairwise judging*: an LLM judge compares a target's transcript with a reference's on the
  same item and picks the more capable model. Every (target, reference) pair is judged, in both
  shown orders. A Bradley-Terry model over the verdicts with the references **pinned at θ\***
  places each target directly on the θ scale. This protocol uses `task.references` only;
  expanding history does not expand target pairings. Its likelihood is fixed, with no fitted
  judge sensitivity or bias and no separate calibration stage. Existing reference–reference
  judgments have no effect on target estimates; history–history judgments are not added.
- *Direct judging*: an LLM judge rates each model's correct train transcripts from 1 to 10,
  with available reasoning shown and decimal ratings preserved. A linear calibration maps
  mean ratings of references + history to their task-specific HARD-item θ*, then forecasts
  targets from their mean ratings. Judging covers `task.train` and `task.test`.

**Metric.** Mean squared error between θ̂ and θ\* over the targets, per semester and pooled.
Forecasts are on the θ scale by construction, so no shift- or scale-free variant is reported.

## Pipeline

```
uv run collect    1. rollouts of every model on every bench          -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run reference  2. freeze a domain: global 2PL, semester tasks, θ*  -> data/meta/full/{samples,items,models,tasks,task_items}.csv (+ paper subset in data/meta/)
                  3. Task.from_semester: references + historical + targets on shared EASY items
uv run judge      4. collect judgments (JudgeBT / JudgeDirect)        -> logs/judge/<domain>/<semester>/<judge>/<hash>/*.eval
                     scripts/export_judgments.py: the paper judges'  -> data/meta/judgments_{pairwise,direct}.csv
uv run evaluate   5. fit / predict every protocol, MSE vs θ*          -> results/<domain>/eval_<protocol>.csv + leaderboard
```

```bash
uv sync                                               # installs src/ as packages; run everything from the repo root; source .env for OPENROUTER_API_KEY, RUNPOD, HF keys
uv run pytest

# 1. collect: benches × models × repeats (validate with --limit 5 --log-dir /tmp/smoke first; real runs cost money)
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2

# 2. freeze a domain once its collection is complete (refit with --force, then commit data/meta)
uv run reference configs/domain/science.yml

# 4. collect a judge's verdicts / ratings (costs money); resumes, a finished run is only read
uv run judge configs/domain/science.yml --judge configs/judges/gemma4-31b.yml --semester 2026-I

# 5. score protocols (never judges): all four domains and every registered protocol by default
uv run evaluate --protocol pairwise glm_regex --semester 2025-II 2026-I
uv run evaluate configs/domain/science.yml --judge configs/judges/gemma4-31b.yml --semester 2026-I         # one judge's run
uv run evaluate configs/domain/science.yml --direct-judge configs/judges/gemma4-31b.yml --semester 2026-I
```

Older frozen tables may lack the historical labels. Task construction reports and excludes
those models, so fitting uses only the available labels until the frozen data is extended.
Code updates do not modify frozen tables or automatically refit them; prepare and review a
separate data update before treating results as using the full historical training set.

`--direct-judge` accepts one or more existing judge YAMLs, using `DirectJudgeConfig` defaults
and showing reasoning even if a reused pairwise config disables it. It defaults to
`assets/system_prompt/direct.txt`; a domain bench can override this with `direct_system_prompt`.
An explicit `system_prompt` in the judge YAML must be a direct-rating prompt.
Direct logs live in `logs/direct/<domain>/<semester>/<judge>/<condition hash>/`, and
forecasts go to `results/<domain>/eval_direct_judge_<judge>.csv`. Judgments are collected by `uv run judge` only.
`--judge` and `--direct-judge` can be used together; the other protocols still follow `--protocol` (all by default).

With `JudgeDirect(..., include_reasoning=False)`, logs use `direct_reasoning-hidden/` instead
of `direct/` under the configured log directory. The HF judging repository uses these as
separate top-level folders, retaining the domain/semester/judge/condition structure.

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

## Compete: adding a protocol

The task: for each semester task, predict the targets' θ* from their transcripts on the task's train items (EASY items
every model solves, so accuracy says nothing), given the references' θ*. Score: MSE of θ̂ vs θ* over the 64 targets of
the paper's main tasks (2025-II and 2026-I, all four domains); the baseline gives every target the references' mean θ*.
Everything is in `data/` (`data/CONTRACT.md`):

- `data/meta/*.csv`: the benchmark (models, items, tasks, task items with the `train` flag, samples with `.eval` paths);
  `data/meta/full/` adds the off-paper models and targets.
- `data/meta/judgments_pairwise.csv`, `judgments_direct.csv`: every verdict / rating the paper's judges gave, so a
  judge-based protocol needs no API calls. A task has several judges (its top-3 references) and conditions; `core`
  marks the one run per judge the paper scores, `paper` the one behind the headline. Pooling every run hurts.
- `data/meta/forecasts.csv`: the paper's θ̂ per target for every protocol and variant, as published (`paper` rows).
- Transcripts: the `.eval` logs under `data/` (`samples.csv` `log`). `glm_regex` also reads `glm_features/`
  (generality-labs/glm_features) and `embedding` reads `transcript-embeddings/` (generality-labs/transcript-embeddings).

The paper's protocols are registered, so `uv run evaluate` recomputes the paper's numbers offline (~2 min) and puts yours
next to them:

```
$ uv run evaluate --protocol pairwise direct glm_regex embedding mean_ref --semester 2025-II 2026-I   # MSE, lower is better
domain     coding  cyber  math  science  main   n   all
protocol
pairwise     0.14   0.26  0.42     0.21  0.26  64  0.26    # Pairwise judge
glm_regex    0.70   1.57  0.76     0.43  0.86  64  0.86    # Deterministic features
embedding    1.69   1.29  0.66     0.40  1.01  64  1.01    # Embedding
direct       1.41   1.48  0.85     0.31  1.01  64  1.01    # Absolute scoring
mean_ref     2.14   1.39  1.32     0.65  1.37  64  1.37    # Baseline (reference mean)
```

`main` is the score (64 targets; check `n`: a NaN forecast is not counted). `pairwise` and `direct` are
`protocols.judges.JudgeTable`: per task, each judge's core run is converted (pairwise: per-bench weighted least squares
on pair logits; direct: a line from mean rating to θ*) and the judge with the lowest leave-one-out error on the
references is used. Variants: `pairwise_calibrated`, `pairwise_bt`, `direct_item_z`; `JudgeTable(kind, variant,
judge=...)` fixes one judge, `.per_judge(task)` shows them all. The paper's back-tests are the other semesters (drop
`--semester`).

A protocol is a `protocols.Protocol`: `__call__(task) -> pd.Series` of θ̂ indexed by target (NaN where it has nothing
to say), by default `fit(task.references)` then `predict(task.test)`. Write it in any file and score it with
`uv run evaluate --protocol pairwise path/to/mine.py:Mine` (built with no arguments); `scripts/example_protocol.py`
(win rate against the references, 0.57) is a complete one to copy. A `Task` holds `domain`, `semester`, `references`
(previous semester, θ = θ*), `historical` (older models, θ = this task's θ*), `test` (targets, θ NaN) and `truth`
(scoring only). A cohort exposes `models` (θ, `path_<bench>` logs), `items` (the train items: bench, item, guess, a, b)
and `transcripts` (model → (bench, item) → the correct first-attempt transcript). Judge only the train items
(AGENTS.md rule 1). Forecasts go to `results/<domain>/eval_<name>.csv`.

See `CLAUDE.md` for the architecture and `AGENTS.md` for operations: collection sweeps,
OpenRouter provider pinning, Runpod, resume semantics, data hygiene.
