#!/bin/bash
#SBATCH --job-name=Default_Train
#SBATCH --account=<your_allocation>
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Default_Train_%j.out
#SBATCH --error=Default_Train_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/baselines/UNet"
cd "$UNET_DIR"

# Default pipeline: L1 + MSTFT, random split, no Whisper
python3 -u train.py \
    --data "$UNET_DIR/data/train_data_random.npz" \
    --out  "$UNET_DIR/checkpoints_default" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 64 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --percep_w 0.0 \
    --patience 30

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
