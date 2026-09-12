#!/bin/bash
#SBATCH --job-name=Train_UNet
#SBATCH --account=<your_allocation>
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Train_UNet_%j.out
#SBATCH --error=Train_UNet_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/UNet"
cd "$UNET_DIR"

DATA_NPZ="$UNET_DIR/data/train_data_bp.npz"
if [[ ! -f "$DATA_NPZ" ]]; then
  echo "Missing data file: $DATA_NPZ"
  exit 1
fi

python3 -u train_bandpass.py \
    --data "$DATA_NPZ" \
    --out "$UNET_DIR/checkpoints_bp" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 64 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --patience 30

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
