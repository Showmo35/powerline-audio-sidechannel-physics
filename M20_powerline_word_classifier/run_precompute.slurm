#!/bin/bash
#SBATCH --job-name=m20_precomp
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=precomp_%j.out
#SBATCH --error=precomp_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M20_powerline_word_classifier"
python -u precompute.py
