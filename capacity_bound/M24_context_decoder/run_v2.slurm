#!/bin/bash
#SBATCH --job-name=m24_v2
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=m24v2_%j.out
#SBATCH --error=m24v2_%j.err

# M24 improved decoder: bidirectional full-sentence LM rescoring + prosody, on the
# Stage-2 top-100 full-gallery lattice. The decisive open-vocab run.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M24_context_decoder"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -e
python -u decode_v2.py --lattice data/lattice_genM22_big.json --tag genM22_big "$@"
echo "[m24-v2] done"
