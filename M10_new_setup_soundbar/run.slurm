#!/bin/bash
#SBATCH --job-name=sb_eval
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err

set -euo pipefail

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"

cd "<REPO_ROOT>/setup_soundbar"

echo "=== GPU ==="; nvidia-smi -L
echo "=== run ==="
python -u run_eval.py --device cuda "$@"
echo "=== done ==="
