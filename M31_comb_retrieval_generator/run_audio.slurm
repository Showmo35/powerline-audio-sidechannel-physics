#!/bin/bash
#SBATCH --job-name=m31aud
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=aud_%j.out
#SBATCH --error=aud_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M31_comb_retrieval_generator"
set -e
python -u make_audio.py --n 4 --seed 0 --arms raw comb
