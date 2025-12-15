#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate Gaussians4D

python "$1" \
  --source_path "$2" \
  --model_path "$3" \
  --save_iterations "$4" 