#!/bin/bash
#SBATCH --job-name=dust3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate dust3r

python "$1" \
  --model_name "$2" \
  --device "$3" \
  --outdir "$4" \
  --image_size "$5" \
  --filelist "$6"