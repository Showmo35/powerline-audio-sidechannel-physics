#!/bin/bash
#SBATCH --job-name=Prep_UNet
#SBATCH --account=<your_allocation>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=Prep_UNet_%j.out
#SBATCH --error=Prep_UNet_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/baselines/UNet"
cd "$UNET_DIR"

python3 -u prepare_data_bandpass.py \
    --folders May29_Alice \
    --split temporal \
    --out_name train_data_bp.npz

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
