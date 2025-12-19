#!/bin/bash
#SBATCH --job-name=splatt3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate splatt3r

python "$1" \
  --image1 "$2" \
  --outdir "$3" \