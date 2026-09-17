# Frontier models on the coding subset

Collect expensive off-roster models (2026-09: `claude-opus-4.8`, `claude-fable-5`, `gpt-5.6-sol`) on the
coding domain at the smallest cost that still serves the study: only the items a judging protocol and a
θ* fit need, and only with prompt caching verified. Collection only — no refreeze, no roster/meta change.

## Items

Per bench, the **2026-I train set** (the EASY items every protocol judges, AGENTS.md rule 1) **∪ the
2026-I HARD set** (what `reference.freeze.local_fit` scores θ* on), from the frozen
`data/meta/coding/items.csv` (2026-09-15). `agent_bench_os` is skipped.

| bench | train (EASY) | HARD | items | hash dir |
| --- | --- | --- | --- | --- |
| terminal_bench_2_1 | 22 | 60 | 82 | `data/terminal_bench_2_1/0af2ba2c/` |
| swe_bench_mini | 13 | 27 | 40 | `data/swe_bench_mini/2d1e487f/` |

The lists are task arguments of `configs/bench/terminal_bench_2_1_frontier.yml` /
`swe-bench-mini-frontier.yml` (`src/collect/tasks/subset.py`): the hash dir identifies the subset and the
runs are full runs (`--sample_id` logs never count, `data/CONTRACT.md`). Same `name`, ids, prompts and
scorer as the full benches, so both hashes pool as one bench once listed together in a domain yml
(`configs/domain/science.yml` `hle_gold` precedent) — not done yet, see Deferred.

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/items.py --check   # ymls == frozen lists
```

## Caching

Anthropic models cache nothing unless `cache_control` breakpoints reach the API. Through OpenRouter,
inspect 0.3.260 inserts them only on list-form content, and in an agent loop the last message is a
string-content `tool` message: the 2026-08-31 `claude-opus-4.7` terminal run and the 2026-09-17
`claude-opus-4.8` smoke both read back only the ~2.3K static prefix per turn (8% / 6% of prompt
tokens). `probe_cache.py` (2026-09-17, opus-4.8): a marker on the tool message in list form is
honoured and the whole prefix is read back next turn. Fix: `src/collect/tasks/openrouter_compat.py`
(side-effect import in `collect/cli.py`) marks the last message's content, string content converted
to list form. Re-smoke: **93%** of prompt tokens from cache. OpenAI caches automatically
(gpt-5.6-sol smoke: 89%). Gate every full run on a 1-item smoke run:

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/cache_check.py <smoke log.eval>   # PASS >= 50% reads after turn 3
```

## Fable 5 refusals

`claude-fable-5` refuses tasks its safety classifier reads as cyber ("violative cyber content",
`stop_reason: refusal`, OpenRouter `finish_reason: content_filter`); the react agent then loops on
empty turns and the sample scores 0. `probe_refusals.py` (first turn only, so a lower bound):
**10 / 82 terminal tasks** (2 train: vulnerable-secret, crack-7z-hash; 8 HARD: write-compressor,
extract-elf, reshard-c4-data, password-recovery, break-filter-js-from-html, path-tracing-reverse,
protein-assembly, custom-memory-heap-crash), **0 / 40 swe**. `claude-opus-5` refuses 4 of the same
(write-compressor, vulnerable-secret, break-filter-js-from-html, crack-7z-hash); opus-4.8 refused
nothing in its smoke. No fallback is possible through OpenRouter. Decisions 2026-09-17: first skipped, then
collected the same day; **refusals are not scored** (`unscore_refusals.py`: Anthropic `content_filter` stops and
OpenAI cyber 403s lose their score and reduction entry, stamped, `sample.metadata.refusal` set), so refused items
drop out of θ* and judging instead of counting as failures. Note `evals.responses` maps an unscored sample to
`correct=False`: the integration step must filter on `metadata.refusal`.

## Commands

```bash
uv run python scripts/audit_configs.py                        # provider pins, before any paid run
S=/tmp/frontier-smoke   # 1 item per model, terminal bench, then cache_check.py on the logs
MODELS_FILE=configs/collect/frontier_models.txt BENCH=configs/bench/terminal_bench_2_1_frontier.yml \
  JOBS=1 LOG_DIR=$S scripts/collect_terminal_bench.sh --limit 1 --max-subprocesses 1

# full runs: terminal then swe per model, models one after another (podman: sequential,
# --max-subprocesses 1); resumable; log in artifacts/frontier/collect.log
setsid -f bash -c 'src/challenges/frontier/collect.sh configs/models/openai/gpt-5.6-sol.yml \
  configs/models/anthropic/claude-opus-4.8.yml > artifacts/frontier/collect.log 2>&1'
```

Order run 2026-09-17: gpt-5.6-sol first (cheap end-to-end check of the subset dirs); at 75/82 its
last 3 samples sat in 30-min retry loops because OpenRouter's `openai/flex` endpoint was
"temporarily rate-limited upstream" (the standard tier answered). The eval-set identity includes
the provider pin, so switching tiers would restart the run; instead the chain was stopped (its 3
orphaned sandboxes removed by name) and relaunched as opus-4.8 → gpt-5.6-sol, which resumes the
75 completed samples from its `started` log. Watch the OpenRouter key's `limit_remaining` (daily cap).

`--max-subprocesses 1` (needed only while images are *built*) serialises every sandbox exec: opus-4.8
sat at 1/82 for 40 min with 31 idle containers. These images are pulled, so the flag is gone from
`collect.sh`; on a running eval raise it live: `uv run inspect ctl config --max-subprocesses 32`.

After collection: `bash scripts/audit/logs/check.sh data/terminal_bench_2_1 data/swe_bench_mini`,
`scripts/audit/logs/audit_evals.py` on the two hash dirs, `normalize_reasoning.py --write
--summary-models claude-opus-4.8 claude-fable-5` if their reasoning arrives as provider summaries (as
opus-4.7 / sonnet-4.5's did), `report_collection.py` and `audit_providers.py`, then the
`data/collection_runs.md` entry in the same data commit.

## Results 2026-09-17

| model | terminal (82) | swe (40) | cache reads | cost (`report_collection.py`) | notes |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4.8 | .768, 0 errors | .750, 0 errors | 92% / 93% | $51 + $8 | reasoning = provider thought summaries (0.05 chars per billed token): moved to `.summary` with `normalize_reasoning.py --summary-models claude-opus-4.8 --write` (1665 blocks, stamped), as for opus-4.7 |
| gpt-5.6-sol | .787 over 80 scored (2 cyber-403 refusals unscored) | .750, 0 errors | 94% / 91% | $6 + $2 | terminal on `openai/flex` (4 samples cancelled after 6h of 429 loops, re-run on the standard tier and spliced); swe on the standard tier; OpenAI's cyber classifier answers 403 on vulnerable-secret and break-filter-js-from-html (the tasks Fable/Opus-5 refuse) |
| claude-fable-5 | .815 over 65 scored (14 refusals unscored, 2 sandbox timeouts, 1 cancelled) | .825, 0 errors | 92% / 92% | ~$130 + $29 (list) | collected on request after the skip; summaries normalised like opus-4.8 (844 + 288 blocks) |
| stealth/union-alpha | .765 over 68 scored (14 cancelled in sticky 400 loops, unscored) | .775, 0 errors | n/a | free | added 2026-09-17 on request; anonymous "stealth" endpoint, no reasoning param. The endpoint answers HTTP 400 "Provider returned error" on ~7% of calls at random (17 of 226 in the first terminal attempt, not clustered, not size-related); inspect treats 400 as final, so the first attempt tripped `--fail-on-error 0.1`. Sample-level `--retry-on-error` restarts the sample and rarely helps at ~30 calls per sample (17/82 after 104 restarts); `openrouter_compat.py` now retries that exact 400 shape (`raw: 'ERROR'`) at the request level as transient. |

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

## Integration (2026-09-18)

`integrate.py --write` derives `data/meta/coding_frontier/` from `data/meta/coding/` (frozen tables untouched):
the four models become extra **2026-I targets** with θ* from `reference.freeze.local_fit` on their scored 2026-I HARD
responses (a, b, guess pinned to the global fit; unscored/refused samples excluded), the original 2026-I targets'
role blanked, `responses.csv` extended. `configs/domain/coding_frontier.yml` lists both hashes of each bench.

| model | scored / HARD responses | θ* (2026-I refs span -1.22 .. 0.72) |
| --- | --- | --- |
| claude-fable-5 | 105 / 73 | 2.61 ± 0.17 |
| gpt-5.6-sol | 120 / 86 | 2.33 ± 0.15 |
| claude-opus-4.8 | 122 / 87 | 2.29 ± 0.16 |
| union-alpha | 108 / 74 | 2.19 ± 0.16 |

```bash
PYTHONPATH=src .venv/bin/python src/challenges/frontier/integrate.py --write      # rebuild the derived meta
uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I --protocol chars mean_ref   # smoke: n=4
uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I --judge configs/judges/reasoning/<stem>.yml
```

Smoke (n = 4 targets, MSE in θ units): chars 7.14, mean_ref 5.39, mean_frontier 2.94 — every baseline sits far
below the frontier, which is the point of the exercise. Judge runs land under `logs/judge/coding_frontier/2026-I/`;
they pair the 4 targets against the 8 references on the train items each target answered correctly (no
agent_bench_os for them). Always pass `--semester 2026-I`: the 2025-II columns are coding's, unchanged.
