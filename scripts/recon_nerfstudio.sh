#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate nerfstudio

ns-train "$1" \
  --output-dir "$2" \
  --experiment-name results \
  --timestamp results \
  --vis viewer \
  --viewer.quit-on-train-completion True \
  --max-num-iterations "$3" \
  --viewer.websocket-port-default "$4" \
  nerfstudio-data --data "$5" \
  --downscale-factor 1
