#!/bin/bash
#SBATCH --job-name=FastVGGT
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate FastVGGT

python "$1" \
  --data_path "$2" \
  --output_path "$3" \
  --ckpt_path "$4" \
  --plot