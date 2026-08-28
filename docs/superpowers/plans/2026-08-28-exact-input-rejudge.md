# Exact-Input Rejudge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable 2,100-call experiment that repeats frozen judge inputs and their actual historical Inspect-log system messages once, then reports historical-to-current exact-input agreement.

**Architecture:** One dedicated script owns deterministic manifest creation, Inspect execution, provenance/cost preflight, and flattening. It reuses the repository's scorer, model configs, and `Model` context manager, while treating frozen Parquet rows and HF-mirrored `.eval` samples as the only prompt authorities. One test file exercises the non-trivial deterministic and statistical logic.

**Tech Stack:** Python 3.14, pandas/pyarrow, Inspect AI 0.3.259, standard library, pytest, existing small-judge model wrapper.

**Spec:** `docs/superpowers/specs/2026-08-28-exact-input-rejudge-design.md`

## Global Constraints

- Preserve all unrelated dirty-worktree changes; never reset, stash, or overwrite them.
- Source freeze is `judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66`.
- Seed is `20260828`; AIME years pool into one benchmark.
- Skip Ling; exactly one epoch and at most 2,100 paid judgments.
- Exact system-message content comes from referenced HF Inspect logs, never repository prompt text files.
- OpenRouter response replay caching is disabled; exact endpoint pins and served provenance are recorded.
- Stop before $10 total paid cost.

---

### Task 1: Deterministic manifest and validation

**Files:**
- Create: `scripts/verdict_dissection/exact_rejudge.py`
- Create: `scripts/verdict_dissection/test_exact_rejudge.py`

**Interfaces:**
- Produces: `build_manifest(pairs, store, prompt_loader, seed=20260828, per_cell=100) -> pd.DataFrame`
- Produces: `validate_manifest(manifest, store, per_cell=100) -> None`
- Produces: `system_message_from_log(path, sample_id, epoch) -> dict[str, object]`
- Produces CLI: `exact_rejudge.py manifest [--output-dir PATH]`

- [ ] **Step 1: Write failing tests for exclusion, pooled cells, matching, and hashes**

Create compact synthetic pair/store frames with two judges, two benchmark labels including two AIME years, both shown orders, one answer-only transcript, and repeated historical draws. Assert:

```python
manifest = build_manifest(pairs, store, fake_prompt_loader, seed=20260828, per_cell=2)
assert set(manifest.bench) == {"aime", "gpqa"}
assert manifest.groupby(["judge", "bench"]).size().eq(2).all()
assert not manifest.duplicated("source_draw_id").any()
assert not manifest.duplicated(["judge", "bench", "underlying_pair_hash"]).any()
assert manifest.groupby(["bench", "underlying_pair_hash"]).input_hash.nunique().max() == 1
assert manifest.input_hash.eq(manifest.judge_input.map(sha256_text)).all()
assert manifest.system_prompt_hash.eq(manifest.system_prompt.map(sha256_text)).all()
assert not manifest.content_hash_l.isin({answer_only_hash}).any()
validate_manifest(manifest, store, per_cell=2)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q`

Expected: import failure because `exact_rejudge.py` does not exist.

- [ ] **Step 3: Implement the smallest manifest path**

In `exact_rejudge.py`:

```python
SEED = 20260828
PER_CELL = 100
SKIP_JUDGES = {"ling-2.6-flash"}
SOURCE_KEYS = [
    "bench_source", "judge", "item", "model_left", "model_right",
    "epoch_left", "epoch_right", "log_left", "log_right", "judge_log",
    "judge_sample", "judge_epoch", "content_hash_l", "content_hash_r",
]

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def seeded_rank(*parts: object, seed: int = SEED) -> str:
    return sha256_text("\0".join(map(str, (seed, *parts))))
```

Apply `has_reasoning`, pool AIME, select the dominant post-exclusion config, canonicalize the sorted content-hash pair, choose one shown input by cross-judge coverage plus `seeded_rank`, and greedily add it to eligible underfilled cells. Use the full frozen draw key for `source_draw_id`. Extract the referenced sample's system message, usage, and model-call metadata from the `.eval` log and cache log reads in-process with `functools.lru_cache`.

- [ ] **Step 4: Implement fatal manifest validation**

Check exact cell sizes, excluded judges, input/system SHA-256, content-hash membership/reasoning, target mapping, source-draw uniqueness, pair uniqueness within cells, one order per underlying pair across judges, dominant config choice, and resolvable log samples. Write Parquet atomically via a temporary file and `Path.replace`; refuse to replace a non-identical existing manifest.

- [ ] **Step 5: Run tests until green**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q`

Expected: all manifest tests pass.

- [ ] **Step 6: Generate and validate the real manifest**

Run:

```bash
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py manifest
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py validate
```

Expected: 21 cells, 100 rows each, 2,100 rows, zero Ling, zero answer-only hashes, zero duplicated source IDs/pairs, and two observed historical system-message hashes resolved from `.eval` samples.

### Task 2: Preflight, endpoint snapshots, and dry-run cost

**Files:**
- Modify: `scripts/verdict_dissection/exact_rejudge.py`
- Modify: `scripts/verdict_dissection/test_exact_rejudge.py`
- Create at runtime: `judge_results/verdict_dissection/exact_rejudge_20260828/endpoint_snapshots/*.json`
- Create at runtime: `judge_results/verdict_dissection/exact_rejudge_20260828/config_drift.csv`

**Interfaces:**
- Produces: `snapshot_endpoints(output_dir) -> dict[str, dict]`
- Produces: `historical_current_configs(manifest) -> pd.DataFrame`
- Produces: `estimate_cost(manifest, endpoint_json) -> pd.DataFrame`
- Produces CLI: `exact_rejudge.py dry-run`

- [ ] **Step 1: Add failing tests for exact pins and cost totals**

Use tiny endpoint dictionaries and assert a missing configured endpoint, non-false fallback, or non-sole unpinned route raises `ValueError`. Assert token-price arithmetic and total planned calls include smoke only once.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q`

Expected: missing preflight functions.

- [ ] **Step 3: Implement stdlib endpoint snapshots and route checks**

Fetch the six OpenRouter endpoint URLs with `urllib.request`, add `retrieved_utc` and URL, and atomically save the unmodified response object. Validate `deepinfra/bf16`, `siliconflow/fp8`, `novita/bf16`, and `deepinfra/fp8`; validate `o4-mini` has exactly one OpenAI endpoint; validate Ling has zero endpoints. Compare historical header fields from the source `.eval` with current `ModelConfig` fields and write every 21-cell dominant hash/count/share plus drift columns.

- [ ] **Step 4: Implement dry-run and conservative estimate**

Use historical per-sample input/output usage captured in the manifest with current endpoint prices. Print cells, matched-input counts, routes, config mixture, system hashes, smoke/full pending counts, and cost by judge. Run `runpodctl user` only as an authentication check; do not create a pod.

- [ ] **Step 5: Run tests and real dry-run**

Run:

```bash
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py dry-run
```

Expected: exact live routes, response-cache header plan, 2,100 total calls, and conservative cost below $10. Stop here if not.

### Task 3: Resumable Inspect runner

**Files:**
- Modify: `scripts/verdict_dissection/exact_rejudge.py`
- Modify: `scripts/verdict_dissection/test_exact_rejudge.py`

**Interfaces:**
- Produces: `task_from_manifest(rows) -> inspect_ai.Task`
- Produces: `terminal_sample_ids(log_root) -> set[str]`
- Produces CLI: `exact_rejudge.py run --smoke` and `exact_rejudge.py run`

- [ ] **Step 1: Add failing tests for exact Sample construction and resume ledger**

Assert `Sample.input` is object-identical to manifest text, target is correct, metadata contains all keys/hashes, tasks split on system-message hash, and existing no-answer/format-failure outputs are terminal. Assert a duplicate output ID is fatal.

- [ ] **Step 2: Run tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q`

- [ ] **Step 3: Build tasks from the manifest only**

Construct `MemoryDataset` samples and `Task(solver=[system_message(exact_message), generate()], scorer=judge_scorer())`. Group by judge, benchmark, and system hash. Import `judge_config` from the existing judging runner and use `Model` in one `ExitStack` context per judge.

- [ ] **Step 4: Enforce execution controls**

For OpenRouter, merge these generation headers after YAML args:

```python
{"X-OpenRouter-Cache": "false", "X-OpenRouter-Metadata": "enabled"}
```

For Llama, copy the config with `fallback_gpus=[]`. Call `eval_set` with `epochs=1`, `retry_attempts=1`, `retry_on_error=0`, `log_model_api=True`, and `log_dir_allow_dirty=True`. Scan existing logs before each task and submit only IDs with no prior response/usage/uncertain paid state. Never delete or overwrite logs.

- [ ] **Step 5: Run tests**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q`

- [ ] **Step 6: Run the 30-call smoke**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py run --smoke`

Expected: five terminal manifest IDs per judge; logs under judge-separated paths; no duplicate ID; every OpenRouter response indicates no response-cache hit and exposes route metadata; RunPod reports secure A40.

### Task 4: Flattening, statistics, and smoke gate

**Files:**
- Modify: `scripts/verdict_dissection/exact_rejudge.py`
- Modify: `scripts/verdict_dissection/test_exact_rejudge.py`
- Create at runtime: required Parquet/CSV/README artifacts under the experiment directory.

**Interfaces:**
- Produces: `flatten(manifest, log_root) -> pd.DataFrame`
- Produces: `summaries(rejudgments, seed=20260828) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `cost_table(rejudgments) -> pd.DataFrame`
- Produces CLI: `exact_rejudge.py flatten`

- [ ] **Step 1: Add failing statistical tests**

Use a six-row fixture containing agreement, both transition directions, no-answer, and transport failure. Assert missing outputs are primary disagreements, answered-only denominator excludes missing, Wilson endpoints are finite, upper Bayes bound is blank below 0.5, transitions are correct, and the 10,000-draw stratified bootstrap is deterministic.

- [ ] **Step 2: Implement flattening and provenance extraction**

Read every `.eval`, join by `source_draw_id`, parse `judge_scorer` output, Inspect usage, raw `ModelCall.response.openrouter_metadata`, endpoint attempts, cache headers, generation ID, backend fingerprint, error/status, and log path. Keep missing manifest rows. Fail on multiple paid outputs for one ID.

- [ ] **Step 3: Implement summaries and atomic outputs**

Write `rejudgments.parquet`, `summary.csv`, `summary_by_judge.csv`, and `cost.csv`. Use standard-library `NormalDist().inv_cdf(0.975)` for Wilson z; NumPy RNG for the stratified bootstrap; the specified Bayes formulas; and atomic temporary-file replacement without replacing a non-identical successful artifact.

- [ ] **Step 4: Write/update the runtime README**

Record source hashes, exact commands, prompt/input hashes, pins/snapshots, exclusion counts, config drift, cache controls, failures, cost status, caveats, and the required result name. Do not call it a ceiling.

- [ ] **Step 5: Run tests and flatten smoke**

Run:

```bash
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py flatten
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py validate
```

Expected: 30 completed/failed smoke rows, route/cache provenance present, no prompt/input mismatch, and cost consistent with provider usage.

### Task 5: Full run and final verification

**Files:**
- Modify only runtime artifacts under `judge_results/verdict_dissection/exact_rejudge_20260828/`.

**Interfaces:**
- Consumes all prior commands and artifacts.
- Produces the completed 2,100-row result set and final report.

- [ ] **Step 1: Re-run dry-run against smoke-calibrated cost**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py dry-run`

Expected: 30 terminal IDs, 2,070 pending, projected total below $10.

- [ ] **Step 2: Run remaining manifest rows**

Run: `UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py run`

The runner updates actual cost after every cell and stops before a batch that could cross $10.

- [ ] **Step 3: Flatten and validate final artifacts**

Run:

```bash
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py flatten
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run python scripts/verdict_dissection/exact_rejudge.py validate
UV_CACHE_DIR=/tmp/small-judge-uv-cache uv run pytest scripts/verdict_dissection/test_exact_rejudge.py -q
```

Expected: 2,100 manifest/result rows, at most one paid response per row, exact prompt/input hashes, no answer-only content, all required files, and exact actual cost or an explicitly unreconciled RunPod billing amount.

- [ ] **Step 4: Report results and synthesis reconciliation**

Report exact paid cost; completed/failed counts; per-judge and per-benchmark agreement; equal-benchmark macro/CI and Bayes bounds; configuration/provider/backend drift; all artifact paths; code changes; verification commands; and the reconciled ridge/GBM comparison using the same averaging unit.
