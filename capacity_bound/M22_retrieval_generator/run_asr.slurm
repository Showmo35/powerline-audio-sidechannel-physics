#!/bin/bash
#SBATCH --job-name=m22_asr
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=asr_%j.out
#SBATCH --error=asr_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M22_retrieval_generator"
python -u transcribe_m22.py
