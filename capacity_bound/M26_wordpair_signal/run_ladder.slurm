#!/bin/bash
#SBATCH --job-name=m26_ladder
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=m26ladder_%j.out
#SBATCH --error=m26ladder_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M26_wordpair_signal"
python -u classifier_ladder.py
