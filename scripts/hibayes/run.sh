#!/usr/bin/env bash
# Submit the two headline models. This never prepares or replaces input data.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
cd "$repo"
mkdir -p /fast/jyeon/smalljudge_glm_check/condor/logs_headline \
  /fast/jyeon/smalljudge_glm_check/condor/logs
exec condor_submit_bid 100 scripts/hibayes/condor/headline.sub
