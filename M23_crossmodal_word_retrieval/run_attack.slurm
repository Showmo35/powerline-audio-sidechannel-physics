#!/bin/bash
#SBATCH --job-name=m23_atk
#SBATCH --account=<your_allocation>
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=atk_%j.out
#SBATCH --error=atk_%j.err
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/M23_crossmodal_word_retrieval"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
echo "[atk] stage 1: build gallery + sentence lattices (GPU)"
python -u attack_build.py
echo "[atk] stage 2: LLM reconstruction + lift (GPU)"
python -u attack_llm.py
echo "[atk] done"
