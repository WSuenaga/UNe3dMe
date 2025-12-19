#!/bin/bash
#SBATCH --job-name=MoGe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate MoGe

python "$1" \
  -i "$2" \
  -o "$3" \
  --maps \
  --glb \
  --ply