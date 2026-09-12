#!/bin/bash
#SBATCH --job-name=m14_asr_eval
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=asr_eval_73k_%j.out

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen"
python asr_eval_73k.py
