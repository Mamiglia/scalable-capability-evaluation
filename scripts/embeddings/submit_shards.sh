#!/bin/bash
# Submit N parallel GPU embedding jobs for one group of benches, each taking every N-th log:
#   scripts/embeddings/submit_shards.sh N bench [bench ...]
# e.g. scripts/embeddings/submit_shards.sh 4 gpqa ; scripts/embeddings/submit_shards.sh 3 aime2024 aime2025 aime2026
set -euo pipefail
shards=${1:?usage: submit_shards.sh N bench [bench ...]}
shift
cd /home/jyeon/LASR/small-judge
for ((k = 0; k < shards; k++)); do
    condor_submit_bid 100 scripts/embeddings/embed.sub -append "arguments = --bench $* --shard $k/$shards" | tail -1
done
