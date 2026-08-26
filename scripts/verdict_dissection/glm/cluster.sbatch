#!/usr/bin/env bash
#SBATCH --job-name=glm-ablations
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Run from the staging directory on the cluster (it holds wheels/, repo/, logs/):
#   sbatch repo/scripts/verdict_dissection/glm/cluster.sbatch setup
#   sbatch --array=1-492%24 --dependency=afterok:<setup job id> repo/scripts/verdict_dissection/glm/cluster.sbatch grid
# GLM_GRID picks the grid directory (jobs.txt, configs/, fits/); default is the ablation grid:
#   sbatch --array=1-9 --export=ALL,GLM_GRID=judge_results/verdict_dissection/glm_waves repo/scripts/verdict_dissection/glm/cluster.sbatch grid
set -euo pipefail
stage=${1:?usage: cluster.sbatch setup|grid}
staging=$SLURM_SUBMIT_DIR
venv=$staging/venv
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/cache-$SLURM_JOB_ID

case "$stage" in
  setup)
    # offline venv from the staged wheels; pip runs straight from its wheel
    [ -x "$venv/bin/python" ] || /usr/bin/python3.12 -m venv --without-pip "$venv"
    pip_wheel=$(ls "$staging"/wheels/pip-*.whl)
    "$venv/bin/python" "$pip_wheel/pip" install --no-index --find-links "$staging/wheels" \
      -r "$staging/wheels/linux-requirements.txt"
    # hibayes declares dvc and inspect extras it never imports; install it without them
    "$venv/bin/python" "$pip_wheel/pip" install --no-index --no-deps "$staging"/wheels/hibayes-*.whl
    "$venv/bin/python" -c 'import hibayes.cli.full, numpyro, arviz; print("venv ok")'
    ;;
  grid)
    grid_dir=${GLM_GRID:-judge_results/verdict_dissection/glm_ablations}
    job=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$staging/repo/$grid_dir/jobs.txt")
    model=${job%%/*}
    ablation=${job#*/}
    cd "$staging/repo"
    config=$grid_dir/configs/$model/$ablation.yaml
    out=$grid_dir/fits/$model/$ablation
    if [ -f "$out/held_out_$model.csv" ]; then echo "already done: $job"; exit 0; fi
    [ -f "$out/models/$model/inference_data.nc" ] || "$venv/bin/hibayes-full" --config "$config" --out "$out" --no-tui
    "$venv/bin/python" scripts/verdict_dissection/glm/evaluate.py "$config" "$out"
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
echo "$stage complete: ${job:-}"
