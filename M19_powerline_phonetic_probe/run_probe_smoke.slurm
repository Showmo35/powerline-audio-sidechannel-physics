#!/bin/bash
#SBATCH --job-name=m19_smoke
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M19_powerline_phonetic_probe"
python -u probe.py --feature wide --epochs 1 --batch 8 --workers 4 --max-windows 64
