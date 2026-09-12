#!/bin/bash
#SBATCH --job-name=m24_decode
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=m24_%j.out
#SBATCH --error=m24_%j.err

# M24: context-aware noisy-channel reranking of the powerline retrieval candidates.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M24_context_decoder"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
python -u decode.py "$@"
echo "[m24] done"
