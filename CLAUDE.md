# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `AGENTS.md` too** — it is the operational guide (collection sweeps, OpenRouter provider pinning, Runpod lifecycle, resume semantics, known failure modes, data hygiene). This file covers the research framing and the architecture that only becomes clear across several files. Don't duplicate AGENTS.md here.

## Research idea

Benchmark-free capability estimation. Four questions, in order: (1) is there a benchmark-free way to estimate a model's capability? (2) do transcripts carry capability signal beyond correct/incorrect? (3) can an LLM judge recover it? (4) does that work weak-to-strong (judge weaker than the models it rates) and old-to-new (calibrated on last year's models, applied to this year's)?

Following Epoch AI / Artificial Analysis and psychometrics, capability is the latent θ of a 2PL IRT model: `P(model α solves item i) = c_i + (1−c_i)·σ(a_i(θ_α − b_i))` (c_i = guessing floor, 1/n_choices on MCQ). A benchmark score is one estimator of θ — `score_α = mean_i p_{α,i}` over its items — and saturates when the items are easy for everyone. θ is what we care about. (Assumes unidimensional capability; domains are fit separately for this reason.)

1. **Collect** rollouts of many models on benchmark questions (`src/collect/`, Inspect AI → `.eval` logs in `data/`); a *domain* (`configs/domain/`) pools the benches that share one scale.
2. **Freeze the domain** (`uv run reference <domain.yml>`, `src/capability/semesters.py`) → `data/meta/<domain>/{models,items,responses}.csv + fit.yml`, committed in the data submodule. Global 2PL over every model × item × repeat (items: `a`, `b`, `guess`; models: `global_theta`). Then one **semester task** per semester from 2024-I on (`configs/semesters.yml`, 2023 is one semester): the 8 models of the previous semester are the *references*, the 8 of this one the *targets*. EASY = items the 3 best references (global θ) solve on repeat 0, HARD = the rest, and items some top reference never answered (a bench that generation was not run on, e.g. HLE before 2025) are out of the task (blank `<sem>_set`); the **train set** is a fixed subset of EASY (25 per global-b quartile, seeded, item-level so identical for every model). The **local IRT** refits only θ of references + targets on the HARD items with `a, b, guess` fixed to the global fit: `<sem>_theta` is θ\*, the target of every protocol, and it is on the global scale for every semester. Refit only when a domain's collection is complete, with `--force`, and commit (science frozen 2026-09-10; math still collecting; coding roster uncollected).
3. **Protocol** (`src/capability/protocol.py`): a callable `Task -> θ̂`. A task is one (semester, target): the references with their θ\* (anchors), the target, the train items, and `Task.transcripts(model)` = correct repeat-0 transcripts of the train items (no ground truth, no HARD items). Planned families: tabular regression over transcript features, forecasting, pairwise judging (`src/judge/` + `src/ranking/` Bradley-Terry), pointwise judging. `chars` (mean transcript length, affine on the references) is the harness smoke test.
4. **Evaluate** (`uv run evaluate <domain.yml> --protocol <name>`): θ̂ vs θ\* per semester and pooled (`all`), always in θ space: Pearson r (shift/scale free), MSE, and affine-fit MSE (MSE after the best shift+scale). Writes `data/meta/<domain>/eval_<name>.csv` (per-target θ̂, gitignored).

Controls in the judging pipeline: both-order counterbalancing (position bias), `truncate_frac` length-matching, `include_reasoning` on/off. `uv run domain` (1PL difficulty quartiles under `data/meta/domains/`, scratch) still feeds the `judge`/`rank` experiments. Everything under `scripts/analyses/` (token attribution / whitebox dissection, style-capability spoofing, score prediction) is experimental — do not build on it.

## Commands

```bash
uv pip install -e .                                   # install (uv, python ≥3.13)
uv run pytest                                         # all tests
uv run pytest tests/test_dataset.py -q                # one module
uv run pytest tests/test_scorer.py -k parse           # one test by keyword

# collect: benches × models × repeats -> data/<bench>/<hash>/<model>/<repeat>/*.eval
uv run collect -b configs/bench/gpqa.yml -m configs/models/google/gemma4-31b.yml -n 2
uv run collect ... --limit 5 --log-dir /tmp/smoke     # validate first; real runs cost money

# judge: ad-hoc tuple, or pair configs + CLI overrides (the sweep pattern)
uv run judge -b configs/bench/gpqa.yml --major <model.yml> --minor <model.yml> -j <judge.yml>
uv run judge --cfg configs/experiments/pairs/*.yml -j configs/models/google/gemma4-31b.yml \
  -b configs/bench/gpqa.yml --system_prompt assets/system_prompt/mcq.txt \
  --include_reasoning false --log_dir logs/pilot --limit 2

bash scripts/experiment/pilot.sh --limit 2            # reproducible sweeps (resumable)
uv run python scripts/audit_configs.py                # model-config pins vs live OpenRouter; run before real runs
uv run reference configs/domain/science.yml [--force]  # freeze: global 2PL + semester tasks -> data/meta/science/
uv run evaluate configs/domain/science.yml --protocol chars  # score a protocol against the frozen theta*
uv run domain configs/domain/math.yml                 # 1PL quartiles for the judge experiments (scratch)
```

Tests live in `tests/` mirroring `src/` (pytest `pythonpath = src`); `tests/conftest.py` isolates the Runpod session file. No linter is configured. Source `.env` (OpenRouter/Runpod/HF keys) — `configs/__init__` calls `load_dotenv()`.

## Architecture (what spans files)

**Config layer (`src/configs/`)** — draccus dataclasses. `DraccusConfig.from_cli(spec)` accepts an instance, dict, YAML path, or bare name (a *missing* path with a config suffix raises, a bare slug becomes `name`). Unknown kwargs land in `_kwargs` and are forwarded to `eval_set`. `ExperimentConfig` holds judge/major/minor (`ModelConfig`) + benchmark (`TaskConfig`) and coerces strings/paths to configs in `__post_init__`; any field is CLI-overridable (`--include_reasoning false`), unknown flags route to `eval_set` (`judge.cli._split_extra`).

**Two content hashes decide where things live — never bypass them:**
- `configs/paths.py:hash_task_args` — SHA-256[:8] over the task's *resolved* args (config `task_args` + the task function's defaults via `inspect.signature`; `limit`/`epochs` excluded). Data path: `data/<bench name>/<hash>/<model shortname>/<repeat>/`. Task resolution raises instead of falling back (a silent fallback = new hash = re-paying collection). `collect` writes `task_args.json` beside each hash dir.
- `configs/experiment.py:condition_hash` — over system_prompt, include_reasoning, truncate_frac, max_length, require_both_correct, include_reverse. Judge logs: `<log_dir>/<bench>/<judge shortname>/<condition hash>/`, all pairs of one invocation as tasks of a single `eval_set` (resume works; pair is in the task name).

**Model shortname** = `name.split('/')[-1].lower()` → the data dir; variants (`_fp8`, `_none`, `_32k`) are separate models. `is_successful_eval` (status success, no limit/sample_id) gates both collect-resume and which logs `ExperimentConfig.model_evals` will pair.

**Judge dataset (`src/judge/dataset.py`)** — `read_transcripts` streams samples, keeps correct ones (`util.extract_score`: exactly one scorer, `"C"`, HLE dict form handled), renders via `util.extract_transcript` (drops system + first user message, keeps later user turns/tool calls; reasoning blocks become `<THOUGHT_START>…<THOUGHT_END>` or are stripped; DeepSeek inline `<think>` stripped when reasoning off). `build_dataset` pairs repeat-0 transcripts per question id; self-pair (`major == minor`) uses repeat 0 vs 1 of the same log. Sample ids are `<qid>_1` / `<qid>_2` where the suffix is the *target* position — `include_reverse` emits both. MCQ options are shown per transcript because gpqa shuffles them per model. Metadata records `chars_1/2`, `truncated_1/2`.

**Scorer (`src/judge/scorer.py`)** — `parse_judge_output` reads only the last `ANSWER:` line (so quoting a transcript's own `ANSWER: 128` is not a verdict); NOANSWER ≠ INCORRECT. Metrics beyond accuracy: `answer_rate`, `position_bias` (−1 always T1, +1 always T2), `consistent_accuracy` (right in both orders — a position-reader scores 0.5 on accuracy but 0 here), `length_corr` (phi of "1 longer" vs "picked 1"). Pair-level analysis relies on the `_1/_2` id convention.

**Execution (`src/collect/runner.py`, `src/model/`)** — `eval_runner` spawns one child process per job with a queue-backed progress display; per-job stdout goes to a temp log (`tail -f`). `Model(config)` context manager turns a `ModelConfig` into `eval_set` kwargs and, when `runpod.use` (or `--runpod`), acquires/releases a shared vLLM pod (`model/session.py`, ref-counted across processes). Both CLIs use it, so a judge can also be a Runpod model.

**Benchmarks** — `configs/bench/*.yml` name inspect tasks (`inspect_evals/...`, `matharena/...` from the `inspect-matharena` dependency, `inspect_harbor/...` for Terminal-Bench). Each must expose a single scorer. `name` is the data-dir label; `task` the inspect spec. Harbor tasks take a `ref` that defaults to `latest`, so the dataset can move underneath a fixed `task_args` hash — pin a digest if a run needs to be reproducible. `inspect-harbor` pulls `litellm`, which caps `openai<3`; `[tool.uv] override-dependencies` in `pyproject.toml` ignores that cap so inspect-ai's OpenRouter provider keeps its `openai>=3.1` (we never call litellm).

## Conventions

- `src/` is the stable library; experiment helpers in `scripts/`, artifacts in `analysis/` and `judge_results/`. Don't put analysis code in `src/`.
- `scripts/audit/logs/` is the eval-log audit toolkit: `validate_data.py` (structure), `report_collection.py` (per-run accuracy/cost/reasoning), `audit_evals.py` (per-sample defects), `normalize_reasoning.py` (fixes where reasoning is stored), `eval_edits.py` (`stamp_edit` — every log an audit rewrites records that inside the file). The `audit-evals` skill drives them.
- Wrong configs raise; no defensive fallbacks (they silently create new hash dirs or bogus model dirs).
- `data/` is a git submodule synced to Hugging Face — no casual git mutations there; `data/legacy/` is archived and excluded.
- Log collection runs in `data/collection_runs.md`.
- Over-engineering and over-testing are explicitly unwanted.
