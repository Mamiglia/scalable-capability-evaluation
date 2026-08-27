#!/usr/bin/env bash
#SBATCH --job-name=glm-screen
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Run a glm script (default screen.py; GLM_SCRIPT overrides) on the cluster. From the glm staging directory:
#   sbatch repo/scripts/verdict_dissection/glm/screen.sbatch <screen.py arguments...>
# Uses the hibayes venv plus scikit-learn from the clause-network staging packages.
set -euo pipefail
staging=$SLURM_SUBMIT_DIR
packages=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/staging/clause-network-20260825-174409/packages
export PYTHONPATH=$packages OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$staging/repo"
"$staging/venv/bin/python" -c 'import sklearn, pandas, pyarrow, yaml; print("sklearn", sklearn.__version__)'
"$staging/venv/bin/python" "${GLM_SCRIPT:-scripts/verdict_dissection/glm/screen.py}" "$@"
