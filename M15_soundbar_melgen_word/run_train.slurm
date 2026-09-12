#!/bin/bash
#SBATCH --job-name=m15_plfw
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=<your_allocation>
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# usage:  sbatch -M <cluster_name> run_train.slurm [extra train.py args...]
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M15_soundbar_melgen_word"
python -u train.py --resume "$@"
