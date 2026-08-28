# Exact-Input Rejudge Design

## Goal

Estimate each judge's historical-to-current exact-input agreement by sending one fresh judgment for 100 frozen, answer-bearing shown inputs in each available judge x pooled-benchmark cell. The experiment reuses the historical shown transcript pair, order, user input, and system message from the authoritative Inspect logs; it is not a mathematical ceiling.

## Fixed scope

- Source freeze: `judge_results/verdict_dissection/freezes/v1-f722885fd822-a979476c1a66/`.
- Seed: `20260828`.
- Pool `aime2024`, `aime2025`, and `aime2026` into `aime`, matching `judge_verdict_logistic.load_pairs`.
- Judges: `gpt-oss-120b`, `deepseek-r1-0528`, `o4-mini`, `gemma-4-31b-it`, `deepseek-v4-flash-0731`, and `Llama-3.1-8B-Instruct`.
- Skip `ling-2.6-flash`; OpenRouter currently returns no endpoints and its direct Novita route is not the historical route.
- Twenty-one cells, 100 rows per cell, 2,100 paid judgments total including smoke outputs.
- One fresh judgment per manifest row. A second epoch is out of scope without explicit approval.
- Hard budget: stop before total estimated or actual paid cost can exceed $10.

## Authoritative inputs

`pairs.parquet` supplies the exact `judge_input`, historical verdict, rollout provenance, shown order, configuration hash, and content hashes. The runner never reconstructs this user input.

The historical system message is not read from a repository prompt file. For every selected source draw, the manifest builder locates the referenced sample in the HF-mirrored historical `.eval` log under `/Volumes/T7/data/hf-judging`, extracts its `ChatMessageSystem` content, and records the exact content and SHA-256. Selection fails if the referenced log/sample cannot be resolved or if a cell contains more than one system-message hash after restricting to its dominant historical configuration.

The current answer-only exclusion is applied before any grouping or sampling: reject a pair if either transcript content hash maps to a transcript for which `judge.dataset.has_reasoning` is false.

## Deterministic manifest

For each pooled benchmark:

1. Count historical configuration hashes after the answer-only exclusion and choose the largest configuration within every judge x benchmark cell. Record all counts and shares so any mixture is visible.
2. Canonicalize an underlying transcript pair by its sorted two content hashes. Never select both shown orders of that pair.
3. For each underlying pair, choose the available shown order with the broadest eligible cross-judge coverage; break ties with a SHA-256 rank derived from seed, benchmark, pair hash, and input hash.
4. Traverse pairs in a seeded deterministic order, adding the chosen exact input to every eligible underfilled judge cell. This maximizes matched inputs across judges without inspecting transcript content. Continue until every cell has 100 distinct pairs.
5. If several historical draws represent the same judge, configuration, pair, and shown input, choose one by a seeded hash of the complete frozen source-draw key.

Each row gets a stable `source_draw_id` and contains all frozen source keys, original verdict, target (`"1"` when `pick_left == 1`, otherwise `"2"`), content hashes, underlying-pair hash, exact input and input SHA-256, exact system message and SHA-256, seed, historical configuration hash, and current judge route/config snapshot.

Validation is fatal on a missing content hash, answer-only transcript, duplicate source draw, duplicate underlying pair within a cell, second shown order, input-hash mismatch, unresolved Inspect sample, non-uniform system message within a task, wrong cell size, or inclusion of Ling.

## Execution

Use one dedicated script, `scripts/verdict_dissection/exact_rejudge.py`, with small subcommands for manifest generation, validation/dry-run, running, and flattening. Reuse:

- `inspect_ai.dataset.Sample` and `MemoryDataset`;
- `judge.scorer.judge_scorer`;
- `system_message(exact_log_message)` followed by `generate()`;
- `scripts/judge/run_judging.py::judge_config`;
- the existing `Model` context manager and model YAMLs.

Create one Inspect task per judge x benchmark x system-message hash. Every `Sample.input` is the frozen `judge_input` verbatim, and its metadata carries the manifest source keys and hashes.

OpenRouter requests add `X-OpenRouter-Cache: false` and `X-OpenRouter-Metadata: enabled`. Existing endpoint allowlists are validated against the saved live endpoint snapshot before spending; exact endpoint tags and `allow_fallbacks: false` are required where more than one endpoint exists. `o4-mini` is accepted only while the snapshot shows a sole OpenAI endpoint. The RunPod config is constrained to secure A40 with no GPU fallback. Provider-side KV/prefix caching is recorded separately and is not whole-response replay caching.

Set one epoch. Format failures and no-answer outputs are successful completed samples and are never retried. Inspect sample retries are disabled; only provider-library transport handling remains. Before submission, scan all existing experiment logs: any manifest ID with a model response, score, token usage, or uncertain paid status is terminal and is not submitted again. Completed outputs are immutable, new logs use unique paths, and a successful cell is skipped.

The smoke run submits five manifest rows per judge, not five per cell. These 30 results live under the same experiment root and count toward the 2,100 full results; the full run submits only the remaining rows.

## Preflight and cost control

Before smoke:

- save raw OpenRouter endpoint JSON for every remote slug, including the empty Ling response;
- save historical-vs-current configuration comparison from the Inspect log headers and current YAMLs;
- verify the embedded system-message and input hashes;
- verify response caching is explicitly disabled and router metadata enabled;
- show cell counts, matched-input counts, routes, configuration mixtures, planned calls, and estimated cost;
- check RunPod authentication/capacity without provisioning a pod during dry-run.

The full run proceeds sequentially by judge/cell, updating actual usage and cost after each completed log. It stops before the next batch if the conservative remaining-cost estimate plus actual cost exceeds $10. OpenRouter cost comes from per-generation usage. RunPod records pod ID, GPU, billed duration/rate or provider billing entry; estimates remain labelled estimates until reconciled to provider billing.

## Flattening and statistics

Flatten every manifest row exactly once into `rejudgments.parquet`. Missing/failed outputs remain rows with `answered = false` and primary agreement zero. Capture original and repeated verdicts, answered flag, agreement, confidence, explanation, provider, endpoint, model/backend fingerprint, token usage, cost, input and system hashes, log path, and failure status.

`summary.csv` reports judge, benchmark, `n`, answer rate, raw agreement, answered-only agreement, Wilson 95% interval for raw agreement, and answered 1->2 and 2->1 transitions. It also reports the loose Bayes-accuracy lower bound `A` and, only for `A >= 0.5`, upper bound `0.5 + sqrt((A - 0.5) / 2)`.

`summary_by_judge.csv` gives the equal-benchmark macro raw agreement and a benchmark-stratified bootstrap 95% interval. The bootstrap resamples within each benchmark, preserves benchmark sizes, averages benchmark agreements equally, uses seed `20260828`, and runs 10,000 replicates.

`cost.csv` reports actual input/output/cached tokens, request cost, RunPod GPU time/cost, failures, and any unreconciled amount by judge and provider.

## Outputs

Everything lives under `judge_results/verdict_dissection/exact_rejudge_20260828/`:

- `sample_manifest.parquet`
- `endpoint_snapshots/*.json`
- `config_drift.csv`
- `logs/<judge>/.../*.eval`
- `rejudgments.parquet`
- `summary.csv`
- `summary_by_judge.csv`
- `cost.csv`
- `README.md`

The README records exact commands, source freeze hashes, prompt/input hashes, endpoint pins and snapshots, response-cache controls, answer-only exclusions, configuration drift, failures, costs, and the wording `historical-to-current exact-input agreement`.

## Verification

A small test file covers deterministic matched sampling, exclusion, no reverse-order duplication, hash validation, missing-as-disagreement summaries, Wilson intervals, transition counts, Bayes bounds, and stratified bootstrap determinism. Before any paid call, run that test plus manifest validation and a dry-run. After smoke and full execution, re-run validation and assert submitted model-call count and paid-generation count do not exceed one per manifest row.
