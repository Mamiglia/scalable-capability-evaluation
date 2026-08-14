#!/bin/bash
set -euo pipefail

repo=${_CONDOR_JOB_IWD:?HTCondor did not set _CONDOR_JOB_IWD}
venv=${HIBAYES_VENV:-/home/jyeon/.venvs/smalljudge-glm313}
cd "$repo"
export MPLCONFIGDIR=${MPLCONFIGDIR:-${_CONDOR_SCRATCH_DIR:?}/matplotlib}
mkdir -p "$MPLCONFIGDIR"
exec "$venv/bin/python" scripts/hibayes/pipeline.py "$@"
