# Transcript embeddings

`src/protocols/embeddings.py` computes one feature vector per model, the mean Qwen3-Embedding-8B vector of that
model's correct train-item transcripts, and fits a ridge regression of θ* on it (`EmbeddingRegression`, the
`embedding` protocol). `embed.py` builds the vectors it reads.

```bash
# one GPU (bf16); existing files are skipped, so a re-run only fills gaps
uv run python scripts/embeddings/embed.py --bench gpqa arc_challenge hle --out transcript-embeddings
uv run python scripts/embeddings/embed.py --out transcript-embeddings --bench aime2024 aime2025 aime2026 \
  apex_2025 apex_shortlist mmlu_pro_math swe_bench_mini terminal_bench_2_1 agent_bench_os
```

The protocol reads the cache from `transcript-embeddings/` (or `$SCE_EMBEDDINGS`). The released cache is the
`transcript-embeddings` dataset (see the top-level README). The `.npz` array layout is documented in the module
docstring of `src/protocols/embeddings.py`. Channel-filtered variants (`channels-<spec>`) are built by
`src/challenges/channels/embed.py`.
