#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# Build the judge-variation venv offline from pre-staged wheels (compute nodes
# have no internet). Skip-if-done via the .ok marker so requeues are no-ops.
set -euo pipefail

BASE=${BASE:-/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge}
PROJ=$BASE/code/judge_variation

# vllm is NOT installed here: serving reuses code/judge_ft/venv/bin/vllm
# read-only (the mac-downloaded wheel closure lacks linux-only deps).
if [ ! -f "$PROJ/venv/.ok" ]; then
  /usr/bin/python3.12 -m venv "$PROJ/venv"
  source "$PROJ/venv/bin/activate"
  pip install --no-index --find-links "$PROJ/wheels" inspect-ai pyyaml draccus
  touch "$PROJ/venv/.ok"
else
  source "$PROJ/venv/bin/activate"
fi
# idempotent add-ons (safe on rerun): zstd zipfile shim for epoch .eval logs,
# openai client for inspect's openai-api provider (wheel already in vllm closure)
pip install --no-index --find-links "$PROJ/wheels" zipfile-zstd zstandard openai
python -c "import inspect_ai, yaml, draccus, zipfile_zstd, openai; print('imports OK')"
