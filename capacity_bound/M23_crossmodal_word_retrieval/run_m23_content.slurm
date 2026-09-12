#!/bin/bash
#SBATCH --job-name=m23_cont
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=m23c_%j.out
#SBATCH --error=m23c_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M23_crossmodal_word_retrieval"
export M23_VOCAB=content
set -e
echo "[m23-content] stage 1: build content-word mels (GPU)"
python -u build_mels.py
echo "[m23-content] stage 2: retrieval eval (CPU)"
python -u retrieval_eval.py
echo "[m23-content] done"
