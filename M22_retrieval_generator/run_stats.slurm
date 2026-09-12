#!/bin/bash
#SBATCH --job-name=m22_stats
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=stats_%j.out
#SBATCH --error=stats_%j.err

# M22 retrieval statistics: correct-word vs impostor correlation histogram +
# % of test words recovered correctly by nearest-correlation.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M22_retrieval_generator"
set -e
python -u viz_corr_stats.py "$@"
echo "[m22_stats] done"
