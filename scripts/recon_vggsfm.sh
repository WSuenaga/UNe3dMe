#!/bin/bash
#SBATCH --job-name=VGGT
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh

conda activate vggsfm_tmp

python "$1" \
  SCENE_DIR="$2" 

conda deactivate 