#!/bin/bash
#SBATCH --job-name=Cascade_Train
#SBATCH --account=<your_allocation>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Cascade_Train_%j.out
#SBATCH --error=Cascade_Train_%j.err

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

PROJECT="<REPO_ROOT>"
CASCADE_DIR="$PROJECT/Cascade"
E2E_DATA="$PROJECT/E2E/data/e2e_data.npz"
UNET_CKPT="$PROJECT/UNet/checkpoints_bp/best_model.pt"

cd "$CASCADE_DIR"

if [[ ! -f "$E2E_DATA" ]]; then
  echo "Missing data file: $E2E_DATA"
  exit 1
fi
if [[ ! -f "$UNET_CKPT" ]]; then
  echo "Missing UNet checkpoint: $UNET_CKPT"
  exit 1
fi

python3 -u train_cascade.py \
    --data "$E2E_DATA" \
    --unet_ckpt "$UNET_CKPT" \
    --out "$CASCADE_DIR/checkpoints" \
    --device cuda \
    --phase1_epochs 100 \
    --phase2_epochs 60 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --unet_lr_factor 0.1 \
    --alpha 0.5 \
    --beta 1.0 \
    --batch 16 \
    --patience 25 \
    --save_every 20 \
    --decode_batches 8 \
    --num_workers 4 \
    --unet_base_ch 64 \
    --gru_hidden 256 \
    --gru_layers 2 \
    --dropout 0.2

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
