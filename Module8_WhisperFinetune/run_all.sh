#!/bin/bash
#SBATCH --job-name=M8_WhisperFT
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

DIR="<REPO_ROOT>/Module8_WhisperFinetune"
cd "$DIR"

echo "========================================"
echo "Module 8: Whisper Fine-Tuning on Powerline Audio"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

# ── Step 1: Prepare data (already done — skip if npz exists) ─────────────────
echo ""
echo ">>> STEP 1: Check data"
echo "----------------------------------------------------------------------"
if [ -f "$DIR/data/powerline_whisper_data.npz" ]; then
    echo "Data already exists, skipping preparation."
else
    echo "Preparing powerline waveform data..."
    python3 -u prepare_data.py \
        --folders May29_Alice \
        --win_sec 10.0 \
        --hop_sec 2.0 \
        --out_dir "$DIR/data" \
        --out_name powerline_whisper_data.npz \
        --teacher_model base
fi

# ── Step 2: Fine-tune Whisper-small.en (frozen encoder) ──────────────────────
echo ""
echo ">>> STEP 2: Fine-tune Whisper-small.en (encoder frozen, decoder only)"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data     "$DIR/data/powerline_whisper_data.npz" \
    --out      "$DIR/checkpoints" \
    --base_model openai/whisper-small.en \
    --freeze_encoder \
    --epochs   30 \
    --lr       1e-5 \
    --batch    8 \
    --patience 10 \
    --save_every 5 \
    --n_decode_val 64 \
    --num_workers 4

# ── Step 3: Inference on val set ─────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Evaluate (CER / WER on full val set)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data  "$DIR/data/powerline_whisper_data.npz" \
    --ckpt  "$DIR/checkpoints/best_cer_model.pt" \
    --out   "$DIR/inference_output" \
    --batch 8

echo ""
echo "========================================"
echo "Module 8 complete: $(date)"
echo "========================================"
