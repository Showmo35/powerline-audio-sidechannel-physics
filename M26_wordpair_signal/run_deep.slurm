#!/bin/bash
#SBATCH --job-name=m26_deep
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=m26deep_%j.out
#SBATCH --error=m26deep_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M26_wordpair_signal"
python -u deep_highfreq.py --pair which this --n 500 --epochs 40 --nnull 2
