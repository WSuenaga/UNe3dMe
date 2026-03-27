#!/bin/bash
#SBATCH --job-name=mast3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate mast3r

python "$1" \
  --filelist "$2" \
  --outdir "$3" \
  --model_name "$4" \
  --as_pointcloud "$5" \