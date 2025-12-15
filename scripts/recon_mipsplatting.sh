#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate mip-splatting

python "$1" \
  -s "$2" \
  -m "$3" \
  --save_iterations "$4" 