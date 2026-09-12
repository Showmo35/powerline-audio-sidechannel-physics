#!/bin/bash
#SBATCH --job-name=m28_comb
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=m28_%j.out
#SBATCH --error=m28_%j.err

# M28: line-locked per-harmonic mains-comb probe. CPU-only (no GPU requested).
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M28_mains_comb"
export OMP_NUM_THREADS=16
set -e
python -u probe_pair.py "$@"
echo "[m28] done"
