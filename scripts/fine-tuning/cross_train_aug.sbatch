#!/bin/bash
#SBATCH --job-name=jft-cross-aug
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-5
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%A_%a.out

# Rewrite-augmented LOO cells: 3 loo x 2 labels, trained on train_<loo>_<label>_aug.parquet
# (plain + rewrite). Adapters -> cross/<size>/<loo>_<label>_aug/. Skip-if-done.
#   sbatch --export=ALL,JFT_MODEL=Qwen3-0.6B,JFT_OUT=cross/Qwen3-0.6B,JFT_LR=1e-4,JFT_RANK=16 cross_train_aug.sbatch
source hpc_common.sh
CD="$IN/cross_domain"

read cell label < <(python - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import sys
cells = ["loo_gpqa", "loo_aime", "loo_chess"]
labels = ["indomain", "eci"]
combos = [(c, l) for l in labels for c in cells]
print(*combos[int(sys.argv[1])])
EOF
)

A="$OUT/${cell}_${label}_aug/adapters/final"
[ -f "$A/adapter_config.json" ] && { echo "done $cell/$label aug"; exit 0; }
P="$CD/train_${cell}_${label}_aug.parquet"
test -f "$P" || { echo "MISSING $P"; exit 1; }
echo "=== [task $SLURM_ARRAY_TASK_ID] train AUG $cell/$label $(date) ==="
python train.py --pairs "$P" --base-model "$MODEL" \
  --lr "$JFT_LR" --lora-rank "$JFT_RANK" --splits train --out-dir "$A"
echo "=== done $cell/$label aug $(date) ==="
