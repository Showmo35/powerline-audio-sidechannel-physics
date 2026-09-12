#!/bin/bash
#SBATCH --job-name=m26_hf
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=m26hf_%j.out
#SBATCH --error=m26hf_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M26_wordpair_signal"
python -u highfreq_probe.py --pair which this --n 400 --nperm 30
