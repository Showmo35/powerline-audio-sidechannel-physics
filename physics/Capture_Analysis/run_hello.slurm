#!/bin/bash
#SBATCH --job-name=hello_code
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=hello_%j.out
#SBATCH --error=hello_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/physics/Capture_Analysis"
python -u embed_hellos.py
