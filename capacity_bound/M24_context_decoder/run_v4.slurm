#!/bin/bash
#SBATCH --job-name=m24_v4
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=m24v4_%j.out
#SBATCH --error=m24v4_%j.err

# M24 v4: honest joint BEAM-SEARCH decoder (no GT context) + full-sentence rescore
# + lambda sweep, on the Stage-2 top-100 full-gallery lattice.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M24_context_decoder"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
python -u decode_v4.py --lattice data/lattice_genM22_big.json --tag genM22_big "$@"
echo "[m24-v4] done"
