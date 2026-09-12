#!/bin/bash
#SBATCH --job-name=m15_before
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=m15before_%j.out
#SBATCH --error=m15before_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/soundbar_setup/M15_soundbar_melgen_word"
python -u make_m15_before.py
