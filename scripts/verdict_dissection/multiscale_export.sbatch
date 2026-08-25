#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/code/verdict_dissection/freeze-v1-f722885fd8226009/rolling-h32b32-h7168b1024-v1/%x-%j.out
set -euo pipefail

MODE=${1:?usage: sbatch multiscale_export.sbatch aggregate|tokens|tables|prefix_tables}
BASE=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge
FREEZE=$BASE/code/verdict_dissection/freeze-v1-f722885fd8226009
RUN=$FREEZE/rolling-h32b32-h7168b1024-v1
CODE=$RUN/export-repo/scripts/verdict_dissection
OUT=$FREEZE/dataframes-rolling-h32b32-h7168b1024-v1
PY=$BASE/code/judge_interp/venv/bin/python

case "$MODE" in
  aggregate)
    "$PY" "$CODE/aggregate_surprisal.py" \
      "$FREEZE/transcript_store.parquet" "$FREEZE/verdicts_model_90a626e872be.parquet" \
      "$FREEZE/surprisal_cache_content" "$OUT" \
      --rolling-cache "$FREEZE/rolling_surprisal_h32b32_h7168b1024_v1"
    ;;
  tokens)
    "$PY" "$CODE/export_token_predictability.py" \
      "$FREEZE/transcript_store.parquet" \
      "$FREEZE/rolling_surprisal_h32b32_h7168b1024_v1" \
      "$OUT/raw/token-predictability-rolling" --rolling
    ;;
  tables)
    "$PY" "$CODE/export_feature_tables.py" \
      "$FREEZE/transcript_store.parquet" "$FREEZE/verdicts_model_90a626e872be.parquet" \
      "$OUT/raw_features_with_surprisal.parquet" "$OUT" \
      --content-hash f722885fd8226009cdf252e89092ee67a1294636f8318597aab7040e3198322d \
      --contract-hash 90a626e872be2f06da1e03d4e3c4d8e3469e72b01bf31f725d7193f42bf581fb
    ;;
  prefix_tables)
    "$PY" "$CODE/export_feature_tables.py" \
      "$FREEZE/transcript_store.parquet" "$FREEZE/verdicts_model_90a626e872be.parquet" \
      "$FREEZE/dataframes-90a626e872be/raw_features_with_surprisal.parquet" \
      "$FREEZE/dataframes-90a626e872be" \
      --content-hash f722885fd8226009cdf252e89092ee67a1294636f8318597aab7040e3198322d \
      --contract-hash 90a626e872be2f06da1e03d4e3c4d8e3469e72b01bf31f725d7193f42bf581fb
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
