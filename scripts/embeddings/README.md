# Transcript embeddings

## Purpose

This pipeline turns evaluation transcripts into a capability-prediction protocol.

`src/protocols/embeddings.py` computes one feature vector per model: the mean Qwen3-Embedding-8B
vector of that model's correct train-item transcripts. It then fits a ridge regression of theta*
(a model's estimated capability) on that vector. The result is `EmbeddingRegression`, registered
as the `embedding` protocol (ridge, the upstream default), plus `emb_kr` (kernel ridge with a
cosine kernel, the best configuration by leave-one-model-out CV in the science sweep) and `emb_knn`
(cosine 3-nearest-neighbours, the best by target MSE). `hillclimb.py` reproduces that sweep.

## Files

- `env.sh` / `env.sub`: build the `smalljudge-embed` venv and check Hugging Face access.
- `pull_data.py` / `pull_data.sub`: mirror the benchmark logs from `LASR-G5/benchmarks` to `/fast`.
- `embed.py` / `embed.sh` / `embed.sub`: embed transcripts on GPU into the embedding cache.
- `check_store.py`: validate an embedding cache against its source logs.
- `push.py`: upload the embedding cache to `LASR-G5/transcript-embeddings`.
- `hf_check.py`: read-only check of Hugging Face repo access; can create the embeddings repo.
- `cpu.sub`: generic CPU job (tests, evaluation, push) in the `smalljudge-embed` venv.
- `tests/test_embeddings.py`: unit tests for `src/protocols/embeddings.py`; no network or GPU.

## Paths

- Python environment: `/home/jyeon/.venvs/smalljudge-embed`.
- Benchmark data mirror: `/fast/jyeon/smalljudge_data/benchmarks`, symlinked into the repo as `data/`.
- Embedding cache: `/fast/jyeon/smalljudge_embeddings`. Set with the `SMALL_JUDGE_EMBEDDINGS`
  environment variable, which `src/protocols/embeddings.py` reads as the `EMBEDDINGS` constant.
- Condor logs: under the `condor/` directory inside each root above.
- Hugging Face repos: `LASR-G5/benchmarks` (source logs) and `LASR-G5/transcript-embeddings`
  (embedding cache).
- Model snapshot: `/fast/jyeon/hf_cache` (Qwen3-Embedding-8B; the GPU job reads it offline).

## Order of operations

1. Build the venv:
   `condor_submit_bid 100 scripts/embeddings/env.sub`
2. Mirror the benchmark data:
   `condor_submit_bid 100 scripts/embeddings/pull_data.sub`
3. Point the repo's `data/` symlink at the mirror:
   `rmdir data && ln -s /fast/jyeon/smalljudge_data/benchmarks data`
4. Smoke-test embedding on a small slice:
   `condor_submit_bid 100 scripts/embeddings/embed.sub -append 'arguments = --bench gpqa --limit-logs 2 --limit-samples 8 --out /fast/jyeon/smalljudge_embeddings/smoke'`
5. Embed the default benches (gpqa, arc_challenge, hle):
   `condor_submit_bid 100 scripts/embeddings/embed.sub`
6. Embed the remaining benches:
   `condor_submit_bid 100 scripts/embeddings/embed.sub -append 'arguments = --bench aime2024 aime2025 aime2026 apex_2025 apex_shortlist mmlu_pro_math swe_bench_mini terminal_bench_2_1 agent_bench_os'`
7. Validate the cache:
   `condor_submit_bid 100 scripts/embeddings/cpu.sub -append 'arguments = scripts/embeddings/check_store.py'`
8. Run the tests, evaluate the protocols, and push the cache to Hugging Face: see the comment at
   the top of `cpu.sub` for the exact commands.

## Output format

See the module docstring of `src/protocols/embeddings.py` for the exact array layout of each
`.npz` file. This document does not repeat that table.
