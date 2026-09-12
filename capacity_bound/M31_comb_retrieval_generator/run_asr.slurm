#!/bin/bash
#SBATCH --job-name=m31asr
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=asr_%j.out
#SBATCH --error=asr_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M31_comb_retrieval_generator"
set -e
python -u asr_check.py --n 50 --model small --arm raw
