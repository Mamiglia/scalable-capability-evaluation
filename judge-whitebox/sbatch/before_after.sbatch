#!/bin/bash
#SBATCH --job-name=wb-beforeafter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ceph/mrsic_flogel/public/projects/MoHa_20260120_SwitchChangeDetect_spForGLM/misc/small-judge/logs/%x-%j.out

# Matched-basis before/after reweight (sec-7 analogue): Hungarian-aligns the base and ft SVD
# components and reports which candidates changed usage vs matched-direction. CPU only (no GPU),
# but holds BOTH decomps in RAM (2x33GB for 8B) -> big --mem. Reuses the banked decomps + judging
# screens. Submit from staging/whitebox/code:
#   sbatch --export=ALL,BA_DA=<base decomp>,BA_DB=<ft decomp>,BA_SA=<base judging screen>,BA_SB=<ft judging screen>,BA_TAG=q8b_migration sbatch/before_after.sbatch
source wb_common.sh
echo "=== before_after $BA_TAG $(date) ==="
python before_after.py --decomp-a "$BA_DA" --decomp-b "$BA_DB" \
  --screen-a "$BA_SA" --screen-b "$BA_SB" --tag "$BA_TAG"
echo "=== done $(date) ==="
