#!/bin/bash
#SBATCH --job-name=sb_train
#SBATCH --account=<your_allocation>
#SBATCH --partition=gpu-exp
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --output=outputs/train_%j.out
#SBATCH --error=outputs/train_%j.err

set -euo pipefail

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"

cd "<REPO_ROOT>/setup_soundbar"

echo "=== GPU ==="; nvidia-smi -L
echo "=== train ==="
python -u train_whisper.py --device cuda "$@"
echo "=== done ==="
