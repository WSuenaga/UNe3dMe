#!/bin/bash
#SBATCH --job-name=monst3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate monst3r

python "$1" \
  --input_dir "$2" \
  --output_dir "$3" \
  --seq_name "$4" 