#!/bin/bash
#SBATCH --job-name=m16_asr
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/capacity_bound/M16_asr_on_plf_mels"
python train_asr.py "$@"
