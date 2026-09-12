#!/bin/bash
#SBATCH --job-name=m18_analysis
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M18_vit_ctc_lm"

[ -f outputs/charlm.pt ] || python -u train_lm.py   # from-scratch CharLM (skip if done)
for src in real null gen; do
  python -u decode.py --input $src          # greedy / beam / beam+LM
done
for src in real null gen; do
  python -u llm_agent.py --input $src || echo "[llm:$src] skipped (model unavailable)"
done
python -u summarize.py
