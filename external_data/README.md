# Fetching Epoch benchmark eval logs

The `.eval` transcripts under `external_data/epoch-*/` are Inspect AI logs
downloaded from Epoch's Benchmarking Hub by `src/irt/fetch.py`, driven off a
per-benchmark config in `configs/irt/`.

`fetch.py` is benchmark-agnostic. Adding a benchmark = one config + one CSV; no
code changes. The two manual steps below are the fiddly part.

## Which benchmarks have logs

Only benchmarks **Epoch runs itself** (via Inspect) publish logs. In the CSV
bundle below these are the files **without** an `_external` suffix; the
`_external` files (`hle_external.csv`, `mmlu_external.csv`, `scicode_external.csv`,
…) are imported scores only, with an empty `Logs` column and nothing to fetch.

The `_external` check is necessary but not sufficient: some Epoch-run benchmarks
keep their logs in a **private S3 bucket** and return `403 AccessDenied` even with
a valid token — the answer-sensitive math sets **MATH Level 5** and
**FrontierMath** are like this (their `Logs` URLs point at
`epoch-benchmarks-production-private…`). Confirmed downloadable: GPQA Diamond,
OTIS Mock AIME, SimpleQA Verified, SWE-bench Verified, Chess Puzzles. Quick
check: a benchmark is fetchable only if its `Logs` hosts are *not* `*-private*`.

## Step 1 — the leaderboard CSV

All per-benchmark CSVs ship in one public, CC-BY bundle — no login or WAF:

```
curl -O https://epoch.ai/data/benchmark_data.zip && unzip benchmark_data.zip
```

Copy the benchmark's file to the `csv:` path named in its config, e.g.
`otis_mock_aime_2024_2025.csv` → `external_data/epoch-otis-aime/otis_mock_aime.csv`.
Each has `Model version`, `Logs` (S3 URL), `id`, scores, etc. If `Logs` is empty,
that benchmark has no transcripts — stop here.

## Step 2 — a fresh WAF token

The CDN (`logs.epoch.ai`) sits behind an AWS WAF. `fetch.py` sends an
`aws-waf-token` cookie via `$EPOCH_WAF_TOKEN`:

1. Open any `Log viewer` URL from the CSV (a `logs.epoch.ai/...` link) in a
   browser; clear the CAPTCHA if prompted.
2. DevTools → Application → Cookies → `logs.epoch.ai` → copy the value of
   `aws-waf-token`.
3. `export EPOCH_WAF_TOKEN='<value>'`

The token expires — re-grab it per session. Without it, downloads hit the WAF
CAPTCHA and fail.

## Step 3 — fetch

```
PYTHONPATH=src python3 -m irt.fetch --config_path configs/irt/otis_aime.yaml --limit 2
```

`--limit 2` is a test pull; drop it for the full set. Downloads are idempotent
(only missing `.eval` files are pulled). Sizes vary enormously — GPQA ~22 MB/model,
Chess/OTIS ~7–10 MB/model, but **SWE-bench trajectories run up to ~14 GB for a
single model (~54 GB for the full set)**. Always `--limit 2` and `du -sh` before a
full pull, especially for agentic benchmarks.

**Known issue (py3.14):** `draccus.parse` raises `TypeError: int | None is not
callable` on the `limit` field under Python 3.14, so `python -m irt.fetch` won't
start. Until draccus is bumped (or `limit` is re-typed), drive it by importing
`select_models`/`download` from `irt.fetch` directly, or run under Python ≤3.12.

## Notes

- `.eval` files are gitignored; the leaderboard CSV is committed.
- `configs/irt/*.yaml`: one per benchmark. `models: all` pulls everything with a
  log URL; paste a subset of `Model version` names to trim.
- `src/irt/data.py` flattens GPQA logs into a correctness matrix — it is
  GPQA-specific (`score == "C"`) and does not apply to the other benchmarks.
