#!/bin/bash
#SBATCH --job-name=m14_wrong
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=<your_allocation>
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=wrong_%j.out
#SBATCH --error=wrong_%j.err

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/M14_new_setup_soundbar_melgen"
python -u viz_wrong_grid.py "$@"
