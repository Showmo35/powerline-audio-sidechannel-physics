#!/bin/bash
#SBATCH --job-name=m15_gs
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:58:00
#SBATCH --output=gs_%j.out
#SBATCH --error=gs_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/soundbar_setup/M15_soundbar_melgen_word"
python -u word_candidates_genspace.py
