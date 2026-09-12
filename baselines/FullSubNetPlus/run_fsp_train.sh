#!/bin/bash
#SBATCH --job-name=FSP_Train
#SBATCH --account=<your_allocation>
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --output=FSP_Train_%j.out
#SBATCH --error=FSP_Train_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

PIPE_DIR="<REPO_ROOT>/baselines/FullSubNetPlus"
cd "$PIPE_DIR"

mkdir -p "$PIPE_DIR/checkpoints"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

python3 -u train.py \
    --data "$PIPE_DIR/data/train_data_fullsubnetplus.npz" \
    --out_dir "$PIPE_DIR/checkpoints" \
    --epochs 200 \
    --lr 3e-4 \
    --batch_size 12 \
    --fb_channels 48 \
    --fb_hidden 64 \
    --sb_hidden 96 \
    --subband_size 7 \
    --w_complex_mse 1.0 \
    --w_logmag 0.5 \
    --w_phase 0.2 \
    --w_mrstft 1.0 \
    --patience 25 \
    --save_every 20

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
