#!/bin/bash
#SBATCH --job-name=m24_stage2
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=m24s2_%j.out
#SBATCH --error=m24s2_%j.err

# M24 Stage 2: full-vocab gallery + top-100 candidates (M22 generator), then
# re-run the context-aware noisy-channel decoder on the richer lattice.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M24_context_decoder"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export M23_GEN=m22
set -e
echo "[stage2] build richer lattice (full-vocab gallery + top-100, M22 generator)"
python -u build_lattice.py --tag genM22_big
echo "[stage2] re-decode with context (M24)"
python -u decode.py --lattice data/lattice_genM22_big.json --tag genM22_big
echo "[m24-stage2] done"
