# Transcript embeddings

## Purpose

This pipeline turns evaluation transcripts into a capability-prediction protocol.

`src/protocols/embeddings.py` computes one feature vector per model: the mean Qwen3-Embedding-8B
vector of that model's correct train-item transcripts. It then fits a ridge regression of theta*
(a model's estimated capability) on that vector. The result is `EmbeddingRegression`, registered
as the `embedding` protocol.

## Files

- `embed.py`: embed transcripts on GPU into the embedding cache (see its docstring for the options).

## Paths

- Benchmark logs: `data/` (`generality-labs/benchmarks`).
- Embedding cache: pass it to `embed.py --out`, and point the `SMALL_JUDGE_EMBEDDINGS` environment
  variable at the same directory; `src/protocols/embeddings.py` reads it as the `EMBEDDINGS` constant.
  The published cache is `generality-labs/transcript-embeddings`.
- Model: `Qwen/Qwen3-Embedding-8B` at the revision pinned in `embed.py`.

## Usage

```bash
# smoke test on a small slice
python scripts/embeddings/embed.py --bench gpqa --limit-logs 2 --limit-samples 8 --out <cache>/smoke
# the benches the protocols read
python scripts/embeddings/embed.py --out <cache> --bench gpqa arc_challenge hle aime2024 aime2025 aime2026 \
  apex_2025 apex_shortlist mmlu_pro_math swe_bench_mini terminal_bench_2_1 agent_bench_os
```

Existing files are skipped, so a re-run only fills gaps. `--shard k/n` splits the logs over n parallel GPU jobs.

## Output format

See the module docstring of `src/protocols/embeddings.py` for the exact array layout of each
`.npz` file. This document does not repeat that table.
