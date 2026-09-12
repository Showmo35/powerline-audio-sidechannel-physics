#!/bin/bash
#SBATCH --job-name=M9_UNetWhisper
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/Module9_UNetWhisper"
cd "$DIR"

echo "========================================"
echo "Module 9: Perceptual-Loss UNet → Whisper Fine-Tuning"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

# ── Step 1: Run Module 4 UNet on Module 3 noisy mels ─────────────────────────
echo ""
echo ">>> STEP 1: Enhance noisy mels with Perceptual-Loss UNet (Module 4)"
echo "  Input : Module3 noisy_train/val (5274+590 clips, 80x498 mel)"
echo "  UNet  : Module4_PerceptualLoss/checkpoints/best_model.pt"
echo "----------------------------------------------------------------------"

if [ -f "$DIR/data/unet_whisper_data.npz" ]; then
    echo "Data already exists, skipping."
else
    python3 -u prepare_data.py
fi

# ── Step 2: Fine-tune Whisper on enhanced mels ───────────────────────────────
echo ""
echo ">>> STEP 2: Fine-tune Whisper-small.en on UNet-enhanced mels (full model)"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data       "$DIR/data/unet_whisper_data.npz" \
    --out        "$DIR/checkpoints" \
    --base_model openai/whisper-small.en \
    --epochs     30 \
    --lr         1e-5 \
    --batch      16 \
    --patience   10 \
    --save_every 5 \
    --n_decode_val 64 \
    --num_workers  4

# ── Step 3: Inference ─────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Evaluate (CER / WER on full val set)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data  "$DIR/data/unet_whisper_data.npz" \
    --ckpt  "$DIR/checkpoints/best_cer_model.pt" \
    --out   "$DIR/inference_output" \
    --batch 16

echo ""
echo "========================================"
echo "Module 9 complete: $(date)"
echo "========================================"
