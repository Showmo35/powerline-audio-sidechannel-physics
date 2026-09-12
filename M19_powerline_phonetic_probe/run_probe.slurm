#!/bin/bash
#SBATCH --job-name=m19_probe
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M19_powerline_phonetic_probe"
for feat in env wide; do python -u probe.py --feature $feat "$@"; done
