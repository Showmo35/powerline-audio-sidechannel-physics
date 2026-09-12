#!/bin/bash
#SBATCH --job-name=speech_presence
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=speech_presence_%j.out
#SBATCH --error=speech_presence_%j.err

set -euo pipefail

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"

cd "<REPO_ROOT>/Capture_Analysis"

echo "=== GPU ==="; nvidia-smi -L
echo "=== run ==="
python -u speech_presence.py --device cuda --no-show "$@"
echo "=== done ==="
