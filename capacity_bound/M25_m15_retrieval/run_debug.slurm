#!/bin/bash
#SBATCH --job-name=m25_m15ret
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=m25_%j.out
#SBATCH --error=m25_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M25_m15_retrieval"
python -u build_and_eval.py freq content
