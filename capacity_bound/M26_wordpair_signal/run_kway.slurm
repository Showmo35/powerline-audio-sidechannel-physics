#!/bin/bash
#SBATCH --job-name=m26_kway
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=m26kway_%j.out
#SBATCH --error=m26kway_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M26_wordpair_signal"
python -u kway_scaling.py
