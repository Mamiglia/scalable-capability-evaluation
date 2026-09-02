#!/bin/bash
#SBATCH --job-name=jft-frontier
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Frontier saturation pairs (GPQA + LAB-Bench), judged by base and
# round-1 fine-tuned qwen3-8b.
source hpc_common.sh

[ -f "$OUT/verdicts_frontier_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/verdicts_frontier_base.jsonl"

[ -f "$OUT/verdicts_frontier_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/verdicts_frontier_ft.jsonl"

# full-strength epoch-pool transcripts, held-out items
[ -f "$OUT/verdicts_frontier_epoch_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier_epoch.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/verdicts_frontier_epoch_base.jsonl"

[ -f "$OUT/verdicts_frontier_epoch_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier_epoch.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/verdicts_frontier_epoch_ft.jsonl"

# frontier vs strong-but-not-frontier, epoch transcripts, held-out items
[ -f "$OUT/verdicts_fvs_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier_vs_strong.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/verdicts_fvs_base.jsonl"

[ -f "$OUT/verdicts_fvs_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_frontier_vs_strong.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/verdicts_fvs_ft.jsonl"

echo "=== FRONTIER_DONE $(date) ==="
