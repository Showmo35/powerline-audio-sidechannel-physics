#!/bin/bash
#SBATCH --job-name=m18_llminfer
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M18_vit_ctc_lm"
export HF_HOME=$HOME/.cache/huggingface
python -u llm_infer.py --n 120
