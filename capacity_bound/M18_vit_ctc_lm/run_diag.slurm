#!/bin/bash
#SBATCH --job-name=m18_diag
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M18_vit_ctc_lm"
for src in gen null real; do
  python -u diag.py --input $src --n 400
done
