#!/bin/bash
#SBATCH --job-name=m27_dual
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=m27_%j.out
#SBATCH --error=m27_%j.err
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M27_dualband_retrieval"
python -u train_retrieval.py --resume --steps 30000 --batch 12 --eval-every 2000 "$@"
