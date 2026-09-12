#!/bin/bash
#SBATCH --job-name=Default_Train_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Default_%j.out
#SBATCH --error=Default_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

UNET_DIR="<REPO_ROOT>/UNet"
cd "$UNET_DIR"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# ── Training ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> TRAINING (L1 + MSTFT, random split, no Whisper)"
echo "----------------------------------------------------------------------"

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

# ── Inference ────────────────────────────────────────────────────────────────
echo ""
echo ">>> INFERENCE (val mode: spectrogram comparison + Griffin-Lim audio)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --mode val \
    --ckpt "$UNET_DIR/checkpoints_default/best_model.pt" \
    --out  "$UNET_DIR/inference_output_default" \
    --data "$UNET_DIR/data/train_data_random.npz" \
    --n_samples 10

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
