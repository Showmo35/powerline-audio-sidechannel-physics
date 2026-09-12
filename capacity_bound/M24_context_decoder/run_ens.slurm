#!/bin/bash
#SBATCH --job-name=m24_ens
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:58:00
#SBATCH --output=ens_%j.out
#SBATCH --error=ens_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M24_context_decoder"
mkdir -p outputs
python -u ensemble_3agent.py
