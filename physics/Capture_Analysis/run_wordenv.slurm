#!/bin/bash
#SBATCH --job-name=word_env
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=wordenv_%j.out
#SBATCH --error=wordenv_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/physics/Capture_Analysis"
python -u show_word_envelopes.py
