#!/bin/bash
#SBATCH --job-name=jft-evalxl
#SBATCH --partition=a100,gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Expanded eval judging: base + round-1 adapter verdicts on the enlarged
# eval splits (pairs_xl). Original 1.5k-per-split outputs stay untouched.
source hpc_common.sh

[ -f "$OUT/verdicts_base_xl.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_xl.parquet" --base-model "$MODEL" --tp 1 \
  --splits eval_mm,eval_mixed,eval_items --out "$OUT/verdicts_base_xl.jsonl"

[ -f "$OUT/verdicts_ft_xl.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_xl.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --splits eval_mm,eval_mixed,eval_items \
  --out "$OUT/verdicts_ft_xl.jsonl"

[ -f "$OUT/self_base.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_self.parquet" --base-model "$MODEL" --tp 1 \
  --out "$OUT/self_base.jsonl"

[ -f "$OUT/self_ft.jsonl" ] || \
python pod_verdicts.py --pairs "$IN/pairs_self.parquet" --base-model "$MODEL" --tp 1 \
  --adapter "$OUT/adapters/final" --out "$OUT/self_ft.jsonl"

echo "=== EVALXL_DONE $(date) ==="
