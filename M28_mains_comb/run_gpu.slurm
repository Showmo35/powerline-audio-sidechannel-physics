#!/bin/bash
#SBATCH --job-name=m28_words
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=m28w_%j.out
#SBATCH --error=m28w_%j.err

# M28 analysis #2: full-scale 30-word detection from ALL mains harmonics (GPU).
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M28_mains_comb"
set -e
python -u words_all_harm.py "$@"
echo "[m28-words] done"
