#!/bin/bash
set -euo pipefail

config=${1:?config path required}
output=${2:-${config%.yaml}}
repo=${_CONDOR_JOB_IWD:?HTCondor did not set _CONDOR_JOB_IWD}
venv=${HIBAYES_VENV:-/home/jyeon/.venvs/smalljudge-glm313}
posterior_root=${GLM_POSTERIOR_ROOT:-/fast/jyeon/smalljudge_glm_check/posteriors}
posterior_group=${3:-$(basename "$(dirname "$config")")}

cd "$repo"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${_CONDOR_SCRATCH_DIR:?}/cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${_CONDOR_SCRATCH_DIR:?}/matplotlib}
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

mkdir -p "$output"
source_config_tmp="${_CONDOR_SCRATCH_DIR:?}/source_config.yaml"
source_hashes_tmp="${_CONDOR_SCRATCH_DIR:?}/source_hashes.sha256"
cp "$config" "$source_config_tmp"
mapfile -t inputs < <("$venv/bin/python" -c \
  'import sys, yaml; print(*yaml.safe_load(open(sys.argv[1]))["data_loader"]["paths"]["extracted_data"], sep="\n")' \
  "$config")
sha256sum scripts/hibayes/model.py scripts/hibayes/communicators.py "${inputs[@]}" \
  > "$source_hashes_tmp"

"$venv/bin/hibayes-full" --config "$config" --out "$output" --no-tui

effects="$output/communicate/population_effects.csv"
test -s "$effects"
sha256sum --check "$source_hashes_tmp"
cmp "$config" "$source_config_tmp"
cp "$source_config_tmp" "$output/source_config.yaml.part"
cp "$source_hashes_tmp" "$output/source_hashes.sha256.part"
mv "$output/source_config.yaml.part" "$output/source_config.yaml"
mv "$output/source_hashes.sha256.part" "$output/source_hashes.sha256"

posterior="$output/models/judge_feature_logistic_model/inference_data.nc"
if [[ -f "$posterior" ]]; then
    name=$(basename "$output")
    destination="$posterior_root/$posterior_group/$name"
    mkdir -p "$destination"
    size=$(stat -c %s "$posterior")
    cp "$posterior" "$destination/inference_data.nc.part"
    test "$(stat -c %s "$destination/inference_data.nc.part")" = "$size"
    mv "$destination/inference_data.nc.part" "$destination/inference_data.nc"
    rm -f "$posterior"
fi
