#!/bin/bash
#SBATCH --job-name=m31
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=m31_%x_%j.out
#SBATCH --error=m31_%x_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M31_comb_retrieval_generator"
set -e
python -u train.py --resume --frontend raw "$@"
