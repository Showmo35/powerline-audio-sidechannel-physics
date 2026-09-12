#!/bin/bash
#SBATCH --job-name=M8_WhisperFT_Full
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

DIR="<REPO_ROOT>/baselines/Module8_WhisperFinetune"
cd "$DIR"

echo "========================================"
echo "Module 8b: Whisper Fine-Tuning (FULL model, encoder unfrozen)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

# Data already prepared — skip step 1
echo ""
echo ">>> Data: $DIR/data/powerline_whisper_data.npz (already prepared)"
echo ""

# ── Fine-tune full Whisper (encoder + decoder) ────────────────────────────────
echo ">>> STEP 2: Fine-tune Whisper-small.en (FULL model, no frozen encoder)"
echo "  lr=1e-5, encoder learns to represent powerline audio"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data          "$DIR/data/powerline_whisper_data.npz" \
    --out           "$DIR/checkpoints_unfrozen" \
    --base_model    openai/whisper-small.en \
    --no_freeze_encoder \
    --epochs        30 \
    --lr            1e-5 \
    --batch         8 \
    --patience      10 \
    --save_every    5 \
    --n_decode_val  64 \
    --num_workers   4

# ── Inference ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Evaluate (CER / WER on full val set)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data  "$DIR/data/powerline_whisper_data.npz" \
    --ckpt  "$DIR/checkpoints_unfrozen/best_cer_model.pt" \
    --out   "$DIR/inference_output_unfrozen" \
    --batch 8

echo ""
echo "========================================"
echo "Module 8b complete: $(date)"
echo "========================================"
