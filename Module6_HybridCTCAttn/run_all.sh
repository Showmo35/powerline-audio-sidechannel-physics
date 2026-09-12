#!/bin/bash
#SBATCH --job-name=M6_Hybrid
#SBATCH --account=<your_allocation>
#SBATCH --time=24:00:00
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

DIR="<REPO_ROOT>/Module6_HybridCTCAttn"
cd "$DIR"

echo "========================================"
echo "Module 6: Hybrid CTC/Attention"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ── Step 1: Prepare data ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: Prepare data (bandpass 50-4kHz, UNet predicted mels)"
echo "----------------------------------------------------------------------"

python3 -u prepare_data.py \
    --folders May29_Alice \
    --win_sec 5.0 \
    --hop_sec 1.0 \
    --out_dir "$DIR/data" \
    --out_name transcribe_data.npz \
    --unet_ckpt "<REPO_ROOT>/UNet/checkpoints_bp/best_model.pt"

# ── Step 2: Train hybrid model ──────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Train Hybrid CTC/Attention (0.7 CTC + 0.3 Attn)"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data "$DIR/data/transcribe_data.npz" \
    --out "$DIR/checkpoints" \
    --epochs 200 \
    --lr 1e-3 \
    --batch 16 \
    --d_model 256 \
    --enc_layers 6 \
    --dec_layers 3 \
    --num_heads 4 \
    --conv_kernel 31 \
    --ctc_weight 0.7 \
    --warmup_pct 0.1 \
    --patience 30

# ── Step 3: Inference ────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Evaluate (CTC greedy vs Attention greedy)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data "$DIR/data/transcribe_data.npz" \
    --ckpt "$DIR/checkpoints/best_cer_model.pt" \
    --out "$DIR/inference_output"

echo ""
echo "========================================"
echo "Module 6 complete: $(date)"
echo "========================================"
