#!/bin/bash
#SBATCH --job-name=word_code
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=wordtest_%j.out
#SBATCH --error=wordtest_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/physics/Capture_Analysis"
python -u word_code_test.py "$@"
