#!/bin/bash
#SBATCH --job-name=M4_ChainInfer
#SBATCH --account=<your_allocation>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/baselines/Module4_PerceptualLoss"
cd "$DIR"

echo "========================================"
echo "Module 4 Chained Inference: Perceptual UNet -> Module3 CTC"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

python3 -u infer_chain_ctc.py

echo ""
echo "========================================"
echo "Chain inference complete: $(date)"
echo "========================================"
