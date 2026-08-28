#!/bin/bash
# Verdict mimic: fine-tune one LM per strict held-out fold. Array index = fold.
#   sbatch --array=0-3 scripts/verdict_dissection/mimic/mimic.sbatch <staging_dir> <model_name> [extra train args]
#SBATCH --job-name=verdict-mimic
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%A-%a.out
set -euo pipefail
STAGING=$1
MODEL_NAME=$2
shift 2
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
PYTHON=$BASE/code/judge_ft/venv/bin/python
DESIGN=new-models-and-items-${SLURM_ARRAY_TASK_ID}
OUT=$STAGING/fits/$MODEL_NAME/$DESIGN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[ -f "$OUT/held_out_judge_verdict_from_pair.csv" ] && { echo "done already: $OUT"; exit 0; }
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PYTHON $STAGING/scripts/train_verdict_mimic.py --data-dir $STAGING/data --model-dir $BASE/models/$MODEL_NAME \
    --design $DESIGN --out-dir $OUT "$@"
