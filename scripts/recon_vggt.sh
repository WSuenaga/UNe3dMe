#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate vggt

python "$1" \
  --image-dir "$2" \
  --out-dir "$3" \
  --conf-thres 3.0 \
  --frame-filter All \
  --prediction-mode Pointmap Regression \
  --mode crop \
  --device cuda \
  --show-cam