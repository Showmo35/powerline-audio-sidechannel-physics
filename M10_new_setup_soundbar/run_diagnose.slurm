#!/bin/bash
#SBATCH --job-name=sb_diag
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=outputs/diag_%j.out
#SBATCH --error=outputs/diag_%j.err
set -euo pipefail
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"
cd "<REPO_ROOT>/setup_soundbar"
nvidia-smi -L
python -u diagnose.py --device cuda "$@"
