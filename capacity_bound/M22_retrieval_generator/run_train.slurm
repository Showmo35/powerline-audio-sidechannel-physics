#!/bin/bash
#SBATCH --job-name=m22_retr
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=m22_%j.out
#SBATCH --error=m22_%j.err

# M22: FRESH-trained, open-vocabulary retrieval-optimized powerline→mel generator.
# --resume picks up outputs/last.pt so a requeue continues instead of restarting.
# usage:  sbatch run_train.slurm [extra train_retrieval.py args...]
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/capacity_bound/M22_retrieval_generator"
python -u train_retrieval.py --resume "$@"
