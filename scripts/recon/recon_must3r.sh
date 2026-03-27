#!/bin/bash
#SBATCH --job-name=must3r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# micromamba
eval "$(micromamba shell hook -s bash)"

micromamba activate must3r

python "$1" \
  --image_dir "$2" \
  --output "$3" \
  --weights "ckpt/MUSt3R_512.pth" \
  --retrieval ckpt/MUSt3R_512_retrieval_trainingfree.pth \
  --image_size 512 \
  --file_type glb