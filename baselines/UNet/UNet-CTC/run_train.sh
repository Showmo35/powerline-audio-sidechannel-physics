#!/bin/bash
#SBATCH --job-name=UNet_CTC
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=UNet_CTC_%j.out
#SBATCH --error=UNet_CTC_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "UNet Transcribe: CTC Training on Predicted Mels"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/UNet/transcribe"

python3 -u train.py \
    --data "$PROJECT/UNet/transcribe/data/transcribe_data.npz" \
    --out "$PROJECT/UNet/transcribe/checkpoints" \
    --epochs 200 \
    --batch 16 \
    --lr 3e-4 \
    --gru_hidden 256 \
    --gru_layers 2 \
    --dropout 0.2 \
    --patience 30 \
    --save_every 20 \
    --decode_batches 4 \
    --num_workers 4 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
