#!/bin/bash
#SBATCH --job-name=m31fid
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=fid_%j.out
#SBATCH --error=fid_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M31_comb_retrieval_generator"
set -e
python -u fidelity.py --arms comb raw --n 1500
