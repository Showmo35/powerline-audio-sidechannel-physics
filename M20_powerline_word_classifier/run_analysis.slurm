#!/bin/bash
#SBATCH --job-name=m20_examples
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=examples_%j.out
#SBATCH --error=examples_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M20_powerline_word_classifier"
python -u analysis_examples.py
