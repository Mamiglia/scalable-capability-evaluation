#!/bin/bash
#SBATCH --job-name=jft-cross-train
#SBATCH --partition=gpu_lowp,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-11
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%A_%a.out

# Job array: one task per (cell, label) — 12 combos from cross_domain.yaml.
# Partitions: gpu_lowp,a100 only — the general `gpu` pool has <40GB / no-bf16 cards.
# Skip-if-done; new adapters only.
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=cross/Qwen3-0.6B,JFT_LR=1e-4,JFT_RANK=16 cross_train.sbatch
#   sbatch --export=ALL,JFT_MODEL=Qwen3-8B,JFT_OUT=cross/Qwen3-8B,JFT_LR=1e-4,JFT_RANK=16 cross_train.sbatch
source hpc_common.sh
CD="$IN/cross_domain"

read cell label < <(python - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import sys, yaml
c = yaml.safe_load(open("cross_domain.yaml"))
combos = [(cell, label) for label in c["labels"] for cell in c["cells"]]
i = int(sys.argv[1])
print(combos[i][0], combos[i][1])
EOF
)

A="$OUT/${cell}_${label}/adapters/final"
[ -f "$A/adapter_config.json" ] && { echo "done $cell/$label"; exit 0; }
P="$CD/train_${cell}_${label}.parquet"
test -f "$P" || { echo "MISSING $P"; exit 1; }
echo "=== [task $SLURM_ARRAY_TASK_ID] train $cell/$label lr=$JFT_LR rank=$JFT_RANK $(date) ==="
python train.py --pairs "$P" --base-model "$MODEL" \
  --lr "$JFT_LR" --lora-rank "$JFT_RANK" --splits train --out-dir "$A"
echo "=== done $cell/$label $(date) ==="
