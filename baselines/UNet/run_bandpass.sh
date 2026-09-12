#!/bin/bash
#SBATCH --job-name=BP_Pipeline
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=BP_%j.out
#SBATCH --error=BP_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

UNET_DIR="<REPO_ROOT>/baselines/UNet"
cd "$UNET_DIR"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# ── Data Preparation ─────────────────────────────────────────────────────────
echo ""
echo ">>> DATA PREP (bandpass 50-4000 Hz, random split)"
echo "----------------------------------------------------------------------"

python3 -u prepare_data_bandpass.py \
    --folders May29_Alice \
    --split random \
    --out_name train_data_bp.npz

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> TRAINING (L1 + MSTFT, no Whisper)"
echo "----------------------------------------------------------------------"

python3 -u train_bandpass.py \
    --data "$UNET_DIR/data/train_data_bp.npz" \
    --out  "$UNET_DIR/checkpoints_bp" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 64 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --patience 30

# ── Inference ────────────────────────────────────────────────────────────────
echo ""
echo ">>> INFERENCE (val mode + Whisper transcription)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --mode val \
    --ckpt "$UNET_DIR/checkpoints_bp/best_model.pt" \
    --out  "$UNET_DIR/inference_output_bp" \
    --data "$UNET_DIR/data/train_data_bp.npz" \
    --n_samples 10

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
