#!/bin/bash
#SBATCH --job-name=m15_wifeG
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=wifeG_%j.out
#SBATCH --error=wifeG_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/soundbar_setup/M15_soundbar_melgen_word"
python -u viz_wife_candidates_allgen.py
