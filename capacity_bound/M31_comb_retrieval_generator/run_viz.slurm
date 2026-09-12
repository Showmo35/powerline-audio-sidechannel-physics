#!/bin/bash
#SBATCH --job-name=m31viz
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=viz_%j.out
#SBATCH --error=viz_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M31_comb_retrieval_generator"
set -e
python -u viz_gen_vs_real.py --n 8 --seed 0 --arms raw comb
