#!/bin/bash
#SBATCH --job-name=m20_sameword
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=sameword_%j.out
#SBATCH --error=sameword_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M20_powerline_word_classifier"
WORD=before python -u show_sameword_compare.py
