# Frontier models on the coding subset

Collect expensive off-roster models (`claude-opus-4.8`, `claude-fable-5`, `gpt-5.6-sol`, `stealth/union-alpha`) on
the coding domain at the smallest cost that still serves the study: only the items a judging protocol and a
θ* fit need, and only with prompt caching verified. Collection does not refreeze coding or change its roster.

## Items

Per bench, the **2026-I train set** (the EASY items every protocol judges) **∪ the 2026-I HARD set** (what
`reference.freeze.local_fit` scores θ* on), from the coding freeze (`data/meta/task_items.csv`).
`agent_bench_os` is skipped.

| bench | train (EASY) | HARD | items | hash dir |
| --- | --- | --- | --- | --- |
| terminal_bench_2_1 | 22 | 60 | 82 | `data/terminal_bench_2_1/0af2ba2c/` |
| swe_bench_mini | 13 | 27 | 40 | `data/swe_bench_mini/2d1e487f/` |

The lists are task arguments of `configs/bench/terminal_bench_2_1_frontier.yml` /
`swe-bench-mini-frontier.yml` (`src/collect/tasks/subset.py`): the hash dir identifies the subset and the
runs are full runs (`--sample_id` logs never count, `data/CONTRACT.md`). Same `name`, ids, prompts and
scorer as the full benches, so both hashes pool as one bench once listed together in a domain yml
(`configs/domain/coding_frontier.yml`).

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/items.py --check   # ymls == frozen lists
```

## Caching

Anthropic models cache nothing unless `cache_control` breakpoints reach the API. Through OpenRouter,
inspect 0.3.260 inserts them only on list-form content, and in an agent loop the last message is a
string-content `tool` message, so only the ~2.3K static prefix is read back per turn (6-8% of prompt
tokens). A marker on the tool message in list form is honoured and the whole prefix is read back next turn.
Fix: `src/collect/tasks/openrouter_compat.py` (side-effect import in `collect/cli.py`) marks the last message's
content, string content converted to list form: **93%** of prompt tokens then come from cache. OpenAI caches
automatically (gpt-5.6-sol: 89%). Gate every full run on a 1-item smoke run and check its cache reads.

## Fable 5 refusals

`claude-fable-5` refuses tasks its safety classifier reads as cyber ("violative cyber content",
`stop_reason: refusal`, OpenRouter `finish_reason: content_filter`); the react agent then loops on
empty turns and the sample scores 0. On first turns alone (so a lower bound) it refuses
**10 / 82 terminal tasks** (2 train: vulnerable-secret, crack-7z-hash; 8 HARD: write-compressor,
extract-elf, reshard-c4-data, password-recovery, break-filter-js-from-html, path-tracing-reverse,
protein-assembly, custom-memory-heap-crash) and **0 / 40 swe**. `claude-opus-5` refuses 4 of the same
(write-compressor, vulnerable-secret, break-filter-js-from-html, crack-7z-hash); opus-4.8 refuses
none. No fallback is possible through OpenRouter. **Refusals are not scored** (`unscore_refusals.py`: Anthropic
`content_filter` stops and OpenAI cyber 403s lose their score and reduction entry, stamped, `sample.metadata.refusal`
set), so refused items drop out of θ* and judging instead of counting as failures. Note `evals.responses` maps an
unscored sample to `correct=False`: the integration step filters on `metadata.refusal`.

## Commands

```bash
S=/tmp/frontier-smoke   # 1 item per model, terminal bench; check the cache reads in the logs
MODELS_FILE=configs/collect/frontier_models.txt BENCH=configs/bench/terminal_bench_2_1_frontier.yml \
  JOBS=1 LOG_DIR=$S scripts/collect_terminal_bench.sh --limit 1 --max-subprocesses 1

# full runs: terminal then swe per model, models one after another (podman: sequential); resumable;
# log in artifacts/frontier/collect.log
setsid -f bash -c 'src/challenges/frontier/collect.sh configs/models/openai/gpt-5.6-sol.yml \
  configs/models/anthropic/claude-opus-4.8.yml > artifacts/frontier/collect.log 2>&1'
```

The eval-set identity includes the provider pin, so switching a model to another tier restarts its run.
`--max-subprocesses 1` (needed only while images are *built*) serialises every sandbox exec; these images are
pulled, so `collect.sh` does not pass it. On a running eval, raise it live: `uv run inspect ctl config --max-subprocesses 32`.
Watch the OpenRouter key's `limit_remaining` (daily cap).

After collection: `bash scripts/audit/logs/check.sh data/terminal_bench_2_1 data/swe_bench_mini`,
`scripts/audit/logs/audit_evals.py` on the two hash dirs, `normalize_reasoning.py --write
--summary-models claude-opus-4.8 claude-fable-5` if their reasoning arrives as provider summaries (as
opus-4.7 / sonnet-4.5's did), `report_collection.py`, then the `data/collection_runs.md` entry in the same data commit.

## Results

| model | terminal (82) | swe (40) | cache reads | cost (`report_collection.py`) | notes |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4.8 | .768, 0 errors | .750, 0 errors | 92% / 93% | $51 + $8 | reasoning = provider thought summaries (0.05 chars per billed token): moved to `.summary` with `normalize_reasoning.py --summary-models claude-opus-4.8 --write` (1665 blocks, stamped), as for opus-4.7 |
| gpt-5.6-sol | .787 over 80 scored (2 cyber-403 refusals unscored) | .750, 0 errors | 94% / 91% | $6 + $2 | terminal on `openai/flex` (4 samples cancelled after 429 loops, re-run on the standard tier and spliced); swe on the standard tier; OpenAI's cyber classifier answers 403 on vulnerable-secret and break-filter-js-from-html (the tasks Fable/Opus-5 refuse) |
| claude-fable-5 | .815 over 65 scored (14 refusals unscored, 2 sandbox timeouts, 1 cancelled) | .825, 0 errors | 92% / 92% | ~$130 + $29 (list) | summaries normalised like opus-4.8 (844 + 288 blocks) |
| stealth/union-alpha | .765 over 68 scored (14 cancelled in sticky 400 loops, unscored) | .775, 0 errors | n/a | free | anonymous "stealth" endpoint, no reasoning param. The endpoint answers HTTP 400 "Provider returned error" on ~7% of calls at random; inspect treats 400 as final and sample-level `--retry-on-error` rarely helps at ~30 calls per sample, so `openrouter_compat.py` retries that exact 400 shape (`raw: 'ERROR'`) at the request level as transient. |

Both Anthropic and OpenAI logs keep the same reasoning shape as their predecessors (opus-4.7,
gpt-5): summaries only, so they pair only in the no-reasoning judging arm.

## Cost (estimate, three models, 122 items each; caching working / not)

| model | terminal (82) | swe (40) | total |
| --- | --- | --- | --- |
| claude-opus-4.8 ($5/$25, reads $0.5) | ~$180 / ~$430 | ~$35 / ~$80 | ~$215 / ~$510 |
| claude-fable-5 ($10/$50, reads $1) | ~$360 / ~$860 | ~$70 / ~$160 | ~$430 / ~$1,020 |
| gpt-5.6-sol (flex $1/$5) | ~$20 | ~$5 | ~$25 |

Anchors: opus-4.7 terminal 795K in / 16K out per item; gpt-5 terminal 420K in (88% cached), swe 300K in;
output assumed 3× with thinking on.

## Integration

`integrate.py --write` adds the `frontier` cohort to coding's rows of the long tables (roster rows untouched; read as
domain `coding_frontier`): the four models become extra **2026-I targets** with θ* from `reference.freeze.local_fit`
on their scored 2026-I HARD responses (a, b, guess pinned to the global fit; unscored/refused samples excluded), the
original 2026-I targets' role blanked in that view, their samples added to `samples.csv`.
`configs/domain/coding_frontier.yml` lists both hashes of each bench.

| model | scored / HARD responses | θ* (2026-I refs span -1.22 .. 0.72) |
| --- | --- | --- |
| claude-fable-5 | 105 / 73 | 2.61 ± 0.17 |
| gpt-5.6-sol | 120 / 86 | 2.33 ± 0.15 |
| claude-opus-4.8 | 122 / 87 | 2.29 ± 0.16 |
| union-alpha | 108 / 74 | 2.19 ± 0.16 |
| claude-opus-4.7 (`--extra`, full-bench logs, coding's responses) | — / 87 | 2.11 ± 0.12 |
| gemini-3.5-flash (`--extra`) | — / 87 | 1.77 ± 0.12 |

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/integrate.py --write      # rebuild the derived meta
uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I --protocol chars mean_ref   # smoke: n=4
```

Smoke (n = 4 targets, MSE in θ units): chars 7.14, mean_ref 5.39, mean_frontier 2.94 — every baseline sits far
below the frontier, which is the point of the exercise. Judge runs land under `logs/judge/coding_frontier/2026-I/`;
they pair the targets against the 8 references on the train items each target answered correctly (no
agent_bench_os for them). Always pass `--semester 2026-I`: the 2025-II columns are coding's, unchanged.

## Judging

`judge.sh` judges the targets one at a time, in the given order, with one judge config (the others go into
`exclude_models`, not part of the condition hash, so all passes share one run dir), collecting with `uv run judge`
and scoring with `uv run evaluate`. Use the **reasoning-hidden** kimi config `configs/judges/hidden/kimi-k2-thinking.yml`
(condition 9c1d3abb: reasoning not shown, full train set, no truncation): the frontier transcripts carry summaries or
no reasoning, so the reasoning-on condition would pair none of them, and coding's 2026-I run under 9c1d3abb already
holds the 28 reference-reference pairs, which inspect's generate cache re-serves for free. gpt-5.5 / 5.4 / 5.3 are
skipped here: their only coding transcripts are the Epoch SWE-bench import (a different scaffold, outside the fit).
The `stealth/union-alpha` endpoint was later withdrawn (it was Unbiased's Pareto, `unbiased/pareto`), so it is a
target only, never a judge.

```bash
setsid -f bash -c 'src/challenges/frontier/judge.sh configs/judges/hidden/kimi-k2-thinking.yml \
  claude-fable-5 gpt-5.6-sol claude-opus-4.8 gemini-3.5-flash claude-opus-4.7 union-alpha > artifacts/frontier/judge.log 2>&1'
```

Check the OpenRouter key limit AND the account credit balance first (`/api/v1/auth/key`, `/api/v1/credits`): with
`fail_on_error: false` an exhausted balance turns every verdict into a silent errored sample.

### Results (kimi-k2-thinking, reasoning hidden, n = 6 targets, MSE in θ² against the local-fit θ*)

| protocol | MSE |
| --- | --- |
| judge_kimi-k2-thinking (pinned BT) | **0.88** |
| judgelin_kimi-k2-thinking | 0.94 |
| judgecal_kimi-k2-thinking | 1.07 |
| mean_frontier | 2.53 |
| mean_ref | 4.81 |
| chars | 6.34 |

Per target (θ̂ pinned BT / θ*): opus-4.7 1.18 / 2.11, gemini-3.5-flash 1.54 / 1.77, fable-5 1.63 / 2.61, opus-4.8 0.97 / 2.28,
gpt-5.6-sol 3.48 / 2.34, union-alpha 2.80 / 2.19. The judge underrates every Anthropic transcript by 1-1.3 θ and overrates
gpt-5.6-sol and union-alpha; the within-group ordering is largely wrong (Spearman on 6 points is not worth quoting), the
level is not — every length baseline sits 2-3 θ below the group. 177 logs, 5,886 verdicts, 0 errors, ~$32
(`logs/judge/coding_frontier/2026-I/kimi-k2-thinking/9c1d3abb/`, in generality-labs/judging).

### Prediction scores

Per target m and protocol P: S_m(P) = −log2(e_m(P) / e_m(base)), e_m = (θ̂_m − θ*_m)², base = the constant forecast
at the references' mean θ* (`mean_ref`; with saturated items accuracy carries no information, so under squared error
the no-information forecast is the reference mean). +1 bit = squared error halved. `plot_scores.py` →
`analysis/frontier/prediction_scores.{csv,png,svg}` (gitignored outputs; rerun to regenerate).

| protocol | Fable 5 | GPT-5.6 Sol | Opus 4.8 | Union Alpha | Opus 4.7 | Gemini 3.5 Flash | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matchup linear | 2.50 | 4.29 | 0.84 | 5.77 | 1.94 | 4.13 | **3.24** |
| Matchup BT | 2.78 | 2.01 | 1.55 | 3.62 | 2.31 | 5.74 | 3.00 |
| Matchup calibrated | 1.63 | 5.41 | 1.25 | 4.06 | 1.68 | 2.91 | 2.82 |
| Ridge features | 0.05 | 1.98 | 0.10 | 1.76 | 0.19 | 2.98 | 1.18 |
| Top-3 ref. mean | 0.78 | 0.89 | 0.91 | 0.96 | 1.00 | 1.24 | 0.96 |
| Length (chars) | −0.50 | −0.22 | −0.56 | −0.28 | −0.59 | 0.01 | −0.36 |

Every matchup variant beats the baseline on every target (weakest on Opus 4.8, ≈1 bit); length is worse than the
baseline on five of six; the top-3 reference mean is a flat ≈1 bit because it sits 0.6 θ above the reference mean.

### Four-protocol figure

`plot_scores.py` shows: pairwise judge (`judgelin`, kimi-k2-thinking, reasoning hidden), absolute judge
(`direct_judge`, kimi-k2-thinking, `configs/judges/direct/kimi-k2-thinking.yml`, condition f8860e9c: 54 logs, 1,218
ratings, 1 unrated, in generality-labs/judging), embeddings (`embedding`, Qwen3-Embedding-8B ridge) and deterministic
features (`ridge`). Prediction scores in bits:

| protocol | Fable 5 | GPT-5.6 Sol | Opus 4.8 | Union Alpha | Opus 4.7 | Gemini 3.5 Flash | mean | MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Absolute judge | 3.86 | 6.28 | 2.63 | 6.19 | 4.44 | 2.64 | **4.34** | 0.35 |
| Pairwise judge | 2.50 | 4.29 | 0.84 | 5.77 | 1.94 | 4.13 | 3.24 | 0.94 |
| Deterministic features | 0.05 | 1.98 | 0.10 | 1.76 | 0.19 | 2.98 | 1.18 | 2.99 |
| Embeddings | 2.53 | 1.15 | 1.61 | 0.71 | 1.18 | −0.43 | 1.13 | 2.32 |

Embeddings overshoot every target (θ̂ 3.5-3.9 vs θ* 1.8-2.6): the ridge extrapolates far outside the references' range.

Embedding vectors for the 8 frontier logs were computed with `scripts/embeddings/embed.py` on two Quadro P6000s rather
than the H100s that built the shared cache: torch cu126 (Pascal kernels), fp16 instead of bf16 (`EMBED_DTYPE`), one
unpadded sequence per batch with no attention mask (`--tokens-per-batch 1`) and `EMBED_NO_GQA_SDPA=1` (SDPA's
grouped-query path needs the flash kernel; without it the math kernel materialises the 8192² attention matrix and OOMs
24 GB), `EMBED_TRAIN_ONLY=1 --variant full` (only what the protocol reads: train items, reasoning-included text). Same
model revision and 8192-token chunking as the cache; fp16 vs bf16 is the one numerical difference. These files are
partial (train items only) and are not part of generality-labs/transcript-embeddings.

### GPT-5.x targets

`integrate.py` also makes targets of GPT-5.1 to 5.5 from the Epoch AI SWE-bench import (`EPOCH`: `swe_bench_mini/6d369262`,
SWE-bench only, Epoch's scaffold, θ* from 23 HARD items, sd 0.25-0.32) and of our own-scaffold `gpt-5` (`--extra`). θ*: 5.4
1.48, 5.3-codex 1.29, 5.5 1.19, 5.1 0.85, gpt-5 0.83, 5.2 0.75. Judged with the same kimi conditions (40 pairwise logs + direct
ratings, ~$21; 12 samples of `django__django-12708` exceed the judge's context and stay unscored). Pairing imported logs
needs two things in `src/judge/`: a task-name alias (`swe_bench_verified` = `swe_bench_verified_mini`) and accepting a
prompt that differs only by our "Please solve the following coding issue" prefix (the longer prompt is shown).

Prediction scores (pairwise / absolute / embeddings / features): GPT-5.5* 4.03 / 0.85 / −2.85 / 2.80; GPT-5.4* 0.17 / 6.88 /
−1.76 / 4.96; GPT-5.3 Codex* 1.84 / 1.91 / −2.46 / 6.86; GPT-5.2* 1.87 / −1.82 / −4.59 / 0.15; GPT-5.1* −0.69 / −1.35 /
−4.15 / −1.54; GPT-5 3.07 / 0.60 / −3.58 / 14.32. These targets sit close to the references (baseline error small), so
scores swing widely; 14.32 is a near-exact hit inflated by the log. Embeddings forecast θ ≈ 3.5 for every closed model
and go strongly negative here; the pairwise judge is positive on 10 of 11 targets. The figure (`plot_scores.py`): colour
only, no grid, Union Alpha omitted, right-hand group = the same protocols on the 8 open-weight 2026-I targets (mean, 95% CI
over targets; absolute judge read from the existing coding run `logs/direct/coding/2026-I/kimi-k2-thinking/1a2c14b2`).
