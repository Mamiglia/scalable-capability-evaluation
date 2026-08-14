#!/bin/bash
#SBATCH --job-name=jft-rewrite
#SBATCH --partition=gpu_lowp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Local (vLLM) transcript restyler — no OpenRouter. Rewrites rewrite_sources.parquet
# into neutral / cross-model-voice versions, resumable. Smoke first:
#   sbatch --export=ALL,JFT_MODEL=Qwen3-32B,JFT_OUT=out_qwen06,JFT_LIMIT=24,JFT_RWOUT=rewrites_smoke rewrite.sbatch
# Full:
#   sbatch --export=ALL,JFT_MODEL=Qwen3-32B,JFT_OUT=out_qwen06 rewrite.sbatch
source hpc_common.sh
SRC="${JFT_RWSRC:-rewrite_sources.parquet}"
test -f "$IN/$SRC" || { echo "MISSING $SRC"; exit 1; }

echo "=== rewrite ($MODEL, $SRC) $(date) ==="
python pod_rewrite.py --sources "$IN/$SRC" --base-model "$MODEL" --tp 1 \
  ${JFT_LIMIT:+--limit $JFT_LIMIT} --out "$OUT/${JFT_RWOUT:-rewrites}.jsonl"

echo "=== REWRITE_DONE $(date) ==="
