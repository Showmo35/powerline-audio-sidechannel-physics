#!/bin/bash
#SBATCH --job-name=m18_overfit
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M18_vit_ctc_lm"
python -u overfit_test.py --n 40 --epochs 250
