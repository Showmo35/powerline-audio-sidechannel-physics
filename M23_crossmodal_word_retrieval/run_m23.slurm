#!/bin/bash
#SBATCH --job-name=m23_retr
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=m23_%j.out
#SBATCH --error=m23_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M23_crossmodal_word_retrieval"
set -e
echo "[m23] stage 1: build mels (GPU)"
python -u build_mels.py
echo "[m23] stage 2: retrieval eval (CPU)"
python -u retrieval_eval.py
echo "[m23] done"
