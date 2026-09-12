#!/bin/bash
#SBATCH --job-name=FSP_Full
#SBATCH --account=<your_allocation>
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --output=FSP_Full_%j.out
#SBATCH --error=FSP_Full_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

PIPE_DIR="<REPO_ROOT>/FullSubNetPlus"
cd "$PIPE_DIR"

mkdir -p "$PIPE_DIR/data" "$PIPE_DIR/checkpoints" "$PIPE_DIR/inference_output"

PL_BIN="<REPO_ROOT>/May29_Alice/Chap_4_real.bin"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

echo ""
echo ">>> PREPARE DATA"
echo "----------------------------------------------------------------------"
python3 -u prepare_data.py \
    --folders May29_Alice \
    --preprocess_mode bandpass \
    --window_sec 4.0 \
    --hop_sec 1.0 \
    --train_ratio 0.9 \
    --guard_sec 4.0 \
    --out_dir "$PIPE_DIR/data" \
    --out_name train_data_fullsubnetplus.npz

echo ""
echo ">>> TRAIN"
echo "----------------------------------------------------------------------"
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
echo ">>> EVALUATE"
echo "----------------------------------------------------------------------"
python3 -u evaluate.py \
    --data "$PIPE_DIR/data/train_data_fullsubnetplus.npz" \
    --ckpt "$PIPE_DIR/checkpoints/best_model.pt" \
    --batch_size 8 \
    --out_json "$PIPE_DIR/checkpoints/val_metrics.json"

echo ""
echo ">>> INFERENCE"
echo "----------------------------------------------------------------------"
python3 -u inference.py \
    --pl_bin "$PL_BIN" \
    --chapter_name chapter_04 \
    --ckpt "$PIPE_DIR/checkpoints/best_model.pt" \
    --out_dir "$PIPE_DIR/inference_output" \
    --window_sec 4.0 \
    --hop_sec 1.0 \
    --report_metrics

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
