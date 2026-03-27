#!/bin/bash
#SBATCH --job-name=CUT3R
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate cut3r

python "$1" \
  --inpdir "$2" \
  --outdir "$3" \
  --model_path "$4" \
  --image_size 512 \
  --vis_threshold 1.5 \
  --device cuda