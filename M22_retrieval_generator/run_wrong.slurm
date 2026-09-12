#!/bin/bash
#SBATCH --job-name=m22_wrong
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=wrong_%j.out
#SBATCH --error=wrong_%j.err

# Examples of M22 generated mels that correlate with the WRONG word (failure cases).
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M22_retrieval_generator"
set -e
python -u viz_wrong_grid.py "$@"
echo "[m22_wrong] done"
