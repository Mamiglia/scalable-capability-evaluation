#!/bin/bash
set -euo pipefail

cd "${_CONDOR_JOB_IWD:?HTCondor did not set _CONDOR_JOB_IWD}"
source_files=(scripts/hibayes/*.py scripts/hibayes/*.yaml scripts/hibayes/*.csv
              scripts/hibayes/run.sh scripts/hibayes/condor/*
              scripts/hibayes/sae_assets/*.py scripts/hibayes/sae_assets/*.pt
              scripts/hibayes/sae_assets/*.npz scripts/hibayes/sae_assets/*LICENSE
              tests/test_hibayes_pipeline.py tests/check_hibayes_communicator.py)
protected_data=(analysis/hibayes/gpqa_glm.csv analysis/hibayes/gpqa_correctness.csv)
source_before="${_CONDOR_SCRATCH_DIR:?}/source.before"
analysis_before="${_CONDOR_SCRATCH_DIR:?}/analysis.before"
data_before="${_CONDOR_SCRATCH_DIR:?}/data.before"

echo "=== source revision ==="
git status --short
sha256sum "${source_files[@]}" | tee "$source_before"
sha256sum "${protected_data[@]}" | tee "$data_before"
find analysis/hibayes -type f -printf '%p\t%s\t%T@\n' | sort > "$analysis_before"

echo "=== environments ==="
/usr/bin/python3 -c 'import sys, numpy, pandas, pytest; print(sys.version); print("numpy", numpy.__version__, "pandas", pandas.__version__, "pytest", pytest.__version__)'
/home/jyeon/.venvs/smalljudge-glm313/bin/python -c 'import sys, importlib.metadata as m; print(sys.version); print("hibayes", m.version("hibayes"), "numpyro", m.version("numpyro"))'

echo "=== synthetic tests ==="
/usr/bin/python3 -m pytest -q tests/test_hibayes_pipeline.py
/home/jyeon/.venvs/smalljudge-glm313/bin/python tests/check_hibayes_communicator.py

echo "=== frozen feature parity ==="
/home/jyeon/.venvs/smalljudge-glm/bin/python scripts/hibayes/transcript_features.py \
  --self-check --runs 5 --records 600

sha256sum "${source_files[@]}" > "${_CONDOR_SCRATCH_DIR:?}/source.after"
sha256sum "${protected_data[@]}" > "${_CONDOR_SCRATCH_DIR:?}/data.after"
find analysis/hibayes -type f -printf '%p\t%s\t%T@\n' | sort \
  > "${_CONDOR_SCRATCH_DIR:?}/analysis.after"
cmp "$source_before" "${_CONDOR_SCRATCH_DIR:?}/source.after"
cmp "$data_before" "${_CONDOR_SCRATCH_DIR:?}/data.after"
cmp "$analysis_before" "${_CONDOR_SCRATCH_DIR:?}/analysis.after"
echo "source and analysis artifact manifests unchanged: OK"
