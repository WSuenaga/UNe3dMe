#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate UniK3D

python "$1" \
  --input "$2" \
  --output "$3" \
  --config-file configs/eval/vitl.json \
  --save \
  --save-ply