#!/bin/bash
#SBATCH --job-name=fast3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate fast3r

python "$1" \
  --inpdir "$2" \
  --outdir "$3" \