#!/bin/bash
#SBATCH --job-name=WinT3R
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate wint3r

python "$1" \
  --data_path "$2" \
  --save_dir "$3" \
  --inference_mode offline \
  --ckpt "$4"