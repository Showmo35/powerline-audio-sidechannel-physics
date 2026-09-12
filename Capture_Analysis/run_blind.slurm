#!/bin/bash
#SBATCH --job-name=blind_sp
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=blind_%j.out
#SBATCH --error=blind_%j.err
set -euo pipefail
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"
cd "<REPO_ROOT>/Capture_Analysis"
nvidia-smi -L
python -u speech_presence_blind.py --no-show "$@"
