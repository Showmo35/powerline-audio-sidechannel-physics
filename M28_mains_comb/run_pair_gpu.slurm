#!/bin/bash
#SBATCH --job-name=m28_pgpu
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=m28p_%j.out
#SBATCH --error=m28p_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M28_mains_comb"
set -e
# ALL harmonics on the which/this pair (bridges analyses #1 and #2)
python -u probe_pair_gpu.py --n 300 --K 1600 --T 24 --nperm 100 --tag allharm
# K=200 matched to the running CPU job -> independent implementation cross-check
python -u probe_pair_gpu.py --n 300 --K 200 --T 24 --nperm 100 --tag K200
echo "[m28-pair-gpu] done"
