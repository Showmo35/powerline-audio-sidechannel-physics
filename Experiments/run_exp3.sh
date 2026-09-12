#!/bin/bash
#SBATCH --job-name=Exp3_LargeCTC
#SBATCH --account=<your_allocation>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Exp3_%j.out
#SBATCH --error=Exp3_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "EXP 3: Cascade with larger CTC encoder (512-dim, 3-layer GRU)"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/Cascade"

python3 -u train_cascade.py \
    --data "$PROJECT/E2E/data/e2e_data.npz" \
    --unet_ckpt "$PROJECT/UNet/checkpoints_bp/best_model.pt" \
    --out "$PROJECT/Experiments/exp3_larger_ctc" \
    --phase1_epochs 120 \
    --phase2_epochs 60 \
    --lr 3e-4 \
    --unet_lr_factor 0.1 \
    --alpha 0.5 \
    --beta 1.0 \
    --batch 16 \
    --patience 30 \
    --save_every 20 \
    --decode_batches 8 \
    --num_workers 4 \
    --gru_hidden 512 \
    --gru_layers 3 \
    --dropout 0.2 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
